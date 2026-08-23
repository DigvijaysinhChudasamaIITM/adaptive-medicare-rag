import math

import pytest

from app.models.document import DocumentChunk
from app.rag.confidence import (
    EvidenceConfidenceError,
    compute_evidence_confidence,
)
from app.rag.vector_store import SearchHit

THRESHOLD = 0.7607258856296539


def make_hit(
    *,
    chunk_id: str,
    rank: int,
    score: float,
) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=score,
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            strategy_id="target_416",
            target_tokens=416,
            token_count=50,
            text="Trusted Medicare evidence.",
            heading_context=("Heading",),
            page_numbers=(54,),
            page_start=54,
            page_end=54,
            section_index=1,
            chunk_index=0,
            source_unit_orders=(1,),
        ),
    )


def test_confidence_is_bounded() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    confidence = compute_evidence_confidence(
        retrieved_hits=[hit],
        cited_hits=[hit],
        relevance_threshold=THRESHOLD,
    )

    assert 0.0 <= confidence <= 1.0


def test_confidence_is_deterministic() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    first = compute_evidence_confidence(
        retrieved_hits=[hit],
        cited_hits=[hit],
        relevance_threshold=THRESHOLD,
    )

    second = compute_evidence_confidence(
        retrieved_hits=[hit],
        cited_hits=[hit],
        relevance_threshold=THRESHOLD,
    )

    assert first == second


def test_stronger_top_retrieval_increases_confidence() -> None:
    weaker = make_hit(
        chunk_id="weak",
        rank=1,
        score=0.80,
    )
    stronger = make_hit(
        chunk_id="strong",
        rank=1,
        score=0.90,
    )

    weak_confidence = compute_evidence_confidence(
        retrieved_hits=[weaker],
        cited_hits=[weaker],
        relevance_threshold=THRESHOLD,
    )

    strong_confidence = compute_evidence_confidence(
        retrieved_hits=[stronger],
        cited_hits=[stronger],
        relevance_threshold=THRESHOLD,
    )

    assert strong_confidence > weak_confidence


def test_stronger_cited_evidence_increases_confidence() -> None:
    top = make_hit(
        chunk_id="top",
        rank=1,
        score=0.90,
    )
    weak_citation = make_hit(
        chunk_id="weak",
        rank=2,
        score=0.76,
    )
    strong_citation = make_hit(
        chunk_id="strong",
        rank=2,
        score=0.88,
    )

    weak_confidence = compute_evidence_confidence(
        retrieved_hits=[top, weak_citation],
        cited_hits=[weak_citation],
        relevance_threshold=THRESHOLD,
    )

    strong_confidence = compute_evidence_confidence(
        retrieved_hits=[top, strong_citation],
        cited_hits=[strong_citation],
        relevance_threshold=THRESHOLD,
    )

    assert strong_confidence > weak_confidence


def test_second_strong_source_increases_support_score() -> None:
    first = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.88,
    )
    second = make_hit(
        chunk_id="chunk-2",
        rank=2,
        score=0.88,
    )

    one_source = compute_evidence_confidence(
        retrieved_hits=[first, second],
        cited_hits=[first],
        relevance_threshold=THRESHOLD,
    )

    two_sources = compute_evidence_confidence(
        retrieved_hits=[first, second],
        cited_hits=[first, second],
        relevance_threshold=THRESHOLD,
    )

    assert two_sources > one_source


def test_no_cited_evidence_returns_zero() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    confidence = compute_evidence_confidence(
        retrieved_hits=[hit],
        cited_hits=[],
        relevance_threshold=THRESHOLD,
    )

    assert confidence == 0.0


def test_below_threshold_retrieval_is_rejected() -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.70,
    )

    with pytest.raises(
        EvidenceConfidenceError
    ):
        compute_evidence_confidence(
            retrieved_hits=[hit],
            cited_hits=[hit],
            relevance_threshold=THRESHOLD,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -1.1,
        1.1,
        math.inf,
        math.nan,
    ],
)
def test_invalid_threshold_is_rejected(
    threshold: float,
) -> None:
    hit = make_hit(
        chunk_id="chunk-1",
        rank=1,
        score=0.87,
    )

    with pytest.raises(
        EvidenceConfidenceError
    ):
        compute_evidence_confidence(
            retrieved_hits=[hit],
            cited_hits=[hit],
            relevance_threshold=threshold,
        )


def test_non_finite_cited_score_is_rejected() -> None:
    top = make_hit(
        chunk_id="top",
        rank=1,
        score=0.87,
    )
    invalid = make_hit(
        chunk_id="invalid",
        rank=2,
        score=math.nan,
    )

    with pytest.raises(
        EvidenceConfidenceError
    ):
        compute_evidence_confidence(
            retrieved_hits=[top, invalid],
            cited_hits=[invalid],
            relevance_threshold=THRESHOLD,
        )