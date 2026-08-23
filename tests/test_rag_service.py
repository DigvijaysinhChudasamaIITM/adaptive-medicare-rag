import pytest

from app.clients.openrouter import (
    OpenRouterGenerationResult,
)
from app.models.document import DocumentChunk
from app.models.generation import GeneratedAnswer
from app.rag.citations import CitationIntegrityError
from app.rag.relevance import RelevanceGate
from app.rag.service import (
    RAGService,
    RetrievalServiceError,
)
from app.rag.vector_store import SearchHit

THRESHOLD = 0.7607258856296539


def make_hit(
    *,
    chunk_id: str,
    rank: int,
    score: float,
    page: int = 54,
) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=score,
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            strategy_id="target_416",
            target_tokens=416,
            token_count=50,
            text=(
                f"Trusted Medicare evidence "
                f"for {chunk_id}."
            ),
            heading_context=("Heading",),
            page_numbers=(page,),
            page_start=page,
            page_end=page,
            section_index=1,
            chunk_index=rank - 1,
            source_unit_orders=(rank,),
        ),
    )


class FakeRetriever:
    def __init__(
        self,
        hits: tuple[SearchHit, ...],
    ) -> None:
        self.hits = hits
        self.calls: list[
            tuple[str, int | None]
        ] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[SearchHit, ...]:
        self.calls.append(
            (query, top_k)
        )

        if top_k is None:
            return self.hits

        return self.hits[:top_k]


class FailingRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[SearchHit, ...]:
        raise RuntimeError(
            "embedding backend failed"
        )


class FakeGenerator:
    def __init__(
        self,
        output: GeneratedAnswer,
    ) -> None:
        self.output = output
        self.call_count = 0
        self.evidence_ids: list[str] = []

    async def generate(
        self,
        *,
        question: str,
        evidence: list,
    ) -> OpenRouterGenerationResult:
        self.call_count += 1
        self.evidence_ids = [
            item.chunk_id
            for item in evidence
        ]

        return OpenRouterGenerationResult(
            output=self.output,
            requested_model="test-model",
            returned_model="test-model",
            used_fallback=False,
        )


@pytest.mark.asyncio
async def test_irrelevant_query_never_calls_generator() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.60,
    )

    retriever = FakeRetriever(
        (hit,)
    )
    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Should never be used.",
            citations=["chunk-1"],
        )
    )

    service = RAGService(
        retriever=retriever,
        relevance_gate=RelevanceGate(
            threshold=THRESHOLD
        ),
        generator=generator,
        top_k=10,
        final_top_k=4,
    )

    result = await service.answer(
        "What is the capital of France?"
    )

    assert generator.call_count == 0
    assert result.confidence_score == 0.0
    assert result.sources == []
    assert "enough information" in result.answer


@pytest.mark.asyncio
async def test_relevant_query_calls_generator_once() -> None:
    hits = tuple(
        make_hit(
            chunk_id=f"chunk-{rank}",
            rank=rank,
            score=0.90 - rank * 0.01,
        )
        for rank in range(1, 7)
    )

    retriever = FakeRetriever(
        hits
    )

    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Supported answer.",
            citations=["chunk-1"],
        )
    )

    service = RAGService(
        retriever=retriever,
        relevance_gate=RelevanceGate(
            threshold=THRESHOLD
        ),
        generator=generator,
        top_k=6,
        final_top_k=4,
    )

    result = await service.answer(
        "Supported Medicare question"
    )

    assert generator.call_count == 1
    assert generator.evidence_ids == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]
    assert result.answer == "Supported answer."
    assert len(result.sources) == 1


@pytest.mark.asyncio
async def test_citation_must_be_in_final_evidence() -> None:
    hits = tuple(
        make_hit(
            chunk_id=f"chunk-{rank}",
            rank=rank,
            score=0.90 - rank * 0.01,
        )
        for rank in range(1, 6)
    )

    generator = FakeGenerator(
        GeneratedAnswer(
            answer="Unsupported citation.",
            citations=["chunk-5"],
        )
    )

    service = RAGService(
        retriever=FakeRetriever(hits),
        relevance_gate=RelevanceGate(
            threshold=THRESHOLD
        ),
        generator=generator,
        top_k=5,
        final_top_k=4,
    )

    with pytest.raises(
        CitationIntegrityError
    ):
        await service.answer(
            "Supported Medicare question"
        )


@pytest.mark.asyncio
async def test_source_metadata_is_returned() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.88,
        page=54,
    )

    service = RAGService(
        retriever=FakeRetriever(
            (hit,)
        ),
        relevance_gate=RelevanceGate(
            threshold=THRESHOLD
        ),
        generator=FakeGenerator(
            GeneratedAnswer(
                answer="Supported answer.",
                citations=["chunk-1"],
            )
        ),
    )

    result = await service.answer(
        "Supported Medicare question"
    )

    assert result.sources[0].chunk_id == "chunk-1"
    assert result.sources[0].page_numbers == [54]
    assert result.sources[0].retrieval_rank == 1
    assert (
        result.sources[0].retrieval_score
        == pytest.approx(0.88)
    )
    assert 0.0 < result.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_retrieval_failure_is_wrapped() -> None:
    service = RAGService(
        retriever=FailingRetriever(),
        relevance_gate=RelevanceGate(
            threshold=THRESHOLD
        ),
        generator=FakeGenerator(
            GeneratedAnswer(
                answer="Unused.",
                citations=[],
            )
        ),
    )

    with pytest.raises(
        RetrievalServiceError
    ):
        await service.answer(
            "Medicare question"
        )


def test_final_top_k_cannot_exceed_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="final_top_k",
    ):
        RAGService(
            retriever=FakeRetriever(()),
            relevance_gate=RelevanceGate(
                threshold=THRESHOLD
            ),
            generator=FakeGenerator(
                GeneratedAnswer(
                    answer="Unused.",
                    citations=[],
                )
            ),
            top_k=4,
            final_top_k=5,
        )