import pytest

from app.models.document import DocumentChunk
from app.models.generation import GeneratedAnswer
from app.rag.citations import (
    CitationIntegrityError,
    build_source_evidence,
    validate_citations,
)
from app.rag.prompting import INSUFFICIENT_EVIDENCE_ANSWER
from app.rag.vector_store import SearchHit


def make_hit(
    *,
    chunk_id: str,
    rank: int,
    score: float,
    pages: tuple[int, ...] = (54,),
    text: str = "Trusted Medicare source text.",
) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=score,
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            strategy_id="target_416",
            target_tokens=416,
            token_count=50,
            text=text,
            heading_context=("Test heading",),
            page_numbers=pages,
            page_start=pages[0],
            page_end=pages[-1],
            section_index=1,
            chunk_index=0,
            source_unit_orders=(1, 2),
        ),
    )


def test_valid_single_citation_is_accepted() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=["chunk-1"],
    )

    validated = validate_citations(
        generated,
        [hit],
    )

    assert validated == (hit,)


def test_multiple_citations_preserve_model_order() -> None:
    first = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.88,
    )
    second = make_hit(
        chunk_id="chunk-2",
        rank=2,
        score=0.84,
    )

    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=[
            "chunk-2",
            "chunk-1",
        ],
    )

    validated = validate_citations(
        generated,
        [first, second],
    )

    assert [
        hit.chunk.chunk_id
        for hit in validated
    ] == [
        "chunk-2",
        "chunk-1",
    ]


def test_duplicate_citations_are_deduplicated() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=[
            "chunk-1",
            "chunk-1",
        ],
    )

    validated = validate_citations(
        generated,
        [hit],
    )

    assert validated == (hit,)


def test_invented_citation_fails_closed() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=["fake-chunk-999"],
    )

    with pytest.raises(
        CitationIntegrityError
    ):
        validate_citations(
            generated,
            [hit],
        )


def test_substantive_answer_requires_citation() -> None:
    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=[],
    )

    with pytest.raises(
        CitationIntegrityError
    ):
        validate_citations(
            generated,
            [],
        )


def test_abstention_without_citations_is_allowed() -> None:
    generated = GeneratedAnswer(
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        citations=[],
    )

    validated = validate_citations(
        generated,
        [],
    )

    assert validated == ()


def test_abstention_with_citation_is_rejected() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    generated = GeneratedAnswer(
        answer=INSUFFICIENT_EVIDENCE_ANSWER,
        citations=["chunk-1"],
    )

    with pytest.raises(
        CitationIntegrityError
    ):
        validate_citations(
            generated,
            [hit],
        )


def test_source_metadata_comes_from_search_hit() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=2,
        score=0.846322,
        pages=(54, 55),
        text="Trusted source text.",
    )

    sources = build_source_evidence(
        [hit]
    )

    source = sources[0]

    assert source.chunk_id == "chunk-1"
    assert source.page_numbers == [54, 55]
    assert source.page_start == 54
    assert source.page_end == 55
    assert source.page_reference == "PDF pages 54–55"
    assert source.snippet == "Trusted source text."
    assert source.retrieval_score == pytest.approx(
        0.846322
    )
    assert source.retrieval_rank == 2


def test_non_contiguous_pages_are_preserved() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
        pages=(54, 56),
    )

    sources = build_source_evidence(
        [hit]
    )

    assert sources[0].page_numbers == [54, 56]
    assert (
        sources[0].page_reference
        == "PDF pages 54, 56"
    )


def test_long_snippet_is_bounded_deterministically() -> None:
    text = (
        "Medicare evidence sentence "
        * 50
    )

    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
        text=text,
    )

    sources = build_source_evidence(
        [hit],
        snippet_max_chars=120,
    )

    snippet = sources[0].snippet

    assert len(snippet) <= 120
    assert snippet.endswith("…")


def test_duplicate_retrieved_chunk_ids_fail_closed() -> None:
    first = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )
    duplicate = make_hit(
        chunk_id="chunk-1",
        rank=2,
        score=0.82,
    )

    generated = GeneratedAnswer(
        answer="Supported answer.",
        citations=["chunk-1"],
    )

    with pytest.raises(
        CitationIntegrityError
    ):
        validate_citations(
            generated,
            [first, duplicate],
        )