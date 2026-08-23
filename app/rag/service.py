from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from starlette.concurrency import run_in_threadpool

from app.clients.openrouter import (
    OpenRouterGenerationResult,
)
from app.models.generation import GroundingEvidence
from app.models.grounding import GroundedAnswer
from app.rag.citations import (
    build_source_evidence,
    validate_citations,
)
from app.rag.confidence import (
    compute_evidence_confidence,
)
from app.rag.prompting import (
    INSUFFICIENT_EVIDENCE_ANSWER,
)
from app.rag.relevance import RelevanceDecision
from app.rag.vector_store import SearchHit


class RetrievalServiceError(RuntimeError):
    """Raised when runtime retrieval cannot be completed safely."""


class RAGServiceError(RuntimeError):
    """Raised when grounded orchestration invariants fail."""


class RetrieverLike(Protocol):
    """Minimal retrieval interface required by the RAG service."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[SearchHit, ...]:
        """Retrieve ranked evidence."""


class RelevanceGateLike(Protocol):
    """Minimal relevance-gate interface required by orchestration."""

    def assess(
        self,
        hits: Sequence[SearchHit],
    ) -> RelevanceDecision:
        """Assess whether retrieved evidence is strong enough."""


class GeneratorLike(Protocol):
    """Minimal asynchronous generation interface."""

    async def generate(
        self,
        *,
        question: str,
        evidence: list[GroundingEvidence],
    ) -> OpenRouterGenerationResult:
        """Generate one grounded answer."""


@dataclass(slots=True)
class RAGService:
    """Coordinate retrieval, gating, generation, and source trust."""

    retriever: RetrieverLike
    relevance_gate: RelevanceGateLike
    generator: GeneratorLike
    top_k: int = 10
    final_top_k: int = 4

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if self.final_top_k <= 0:
            raise ValueError(
                "final_top_k must be positive."
            )

        if self.final_top_k > self.top_k:
            raise ValueError(
                "final_top_k cannot exceed top_k."
            )

    async def answer(
        self,
        query: str,
    ) -> GroundedAnswer:
        """Answer one query using only validated retrieved evidence."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "Query must not be empty."
            )

        try:
            hits = await run_in_threadpool(
                self.retriever.retrieve,
                cleaned_query,
                top_k=self.top_k,
            )

            decision = self.relevance_gate.assess(
                hits
            )
        except Exception as exc:
            raise RetrievalServiceError(
                "Retrieval or relevance assessment failed."
            ) from exc

        if not decision.is_relevant:
            return GroundedAnswer(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                confidence_score=0.0,
                sources=[],
            )

        final_hits = tuple(
            hits[: self.final_top_k]
        )

        if not final_hits:
            raise RAGServiceError(
                "Relevant retrieval decision contained no evidence."
            )

        evidence = [
            GroundingEvidence(
                chunk_id=hit.chunk.chunk_id,
                text=hit.chunk.text,
            )
            for hit in final_hits
        ]

        generation = await self.generator.generate(
            question=cleaned_query,
            evidence=evidence,
        )

        cited_hits = validate_citations(
            generation.output,
            final_hits,
        )

        if not cited_hits:
            return GroundedAnswer(
                answer=generation.output.answer,
                confidence_score=0.0,
                sources=[],
            )

        try:
            sources = build_source_evidence(
                cited_hits
            )

            confidence = (
                compute_evidence_confidence(
                    retrieved_hits=hits,
                    cited_hits=cited_hits,
                    relevance_threshold=(
                        decision.threshold
                    ),
                )
            )
        except ValueError as exc:
            raise RAGServiceError(
                "Trusted source enrichment failed."
            ) from exc

        return GroundedAnswer(
            answer=generation.output.answer,
            confidence_score=confidence,
            sources=list(sources),
        )