from collections.abc import Sequence
from math import isfinite

from app.rag.vector_store import SearchHit


class EvidenceConfidenceError(ValueError):
    """Raised when evidence confidence cannot be computed safely."""


def compute_evidence_confidence(
    *,
    retrieved_hits: Sequence[SearchHit],
    cited_hits: Sequence[SearchHit],
    relevance_threshold: float,
) -> float:
    """Compute a bounded deterministic evidence-strength score.

    This score is not a calibrated probability that the generated answer
    is factually correct. It summarizes retrieval and citation evidence
    strength using trusted backend signals.
    """

    _validate_threshold(
        relevance_threshold
    )

    if not cited_hits:
        return 0.0

    if not retrieved_hits:
        raise EvidenceConfidenceError(
            "Retrieved hits are required when cited evidence is present."
        )

    top_score = float(
        retrieved_hits[0].score
    )

    _validate_score(
        top_score,
        name="top retrieval score",
    )

    if top_score < relevance_threshold:
        raise EvidenceConfidenceError(
            "Cannot assign grounded-answer confidence when the "
            "top retrieval score is below the relevance threshold."
        )

    cited_scores = [
        float(hit.score)
        for hit in cited_hits
    ]

    for score in cited_scores:
        _validate_score(
            score,
            name="cited retrieval score",
        )

    absolute_similarity = (
        _normalize_cosine(top_score)
    )

    threshold_margin = (
        _normalize_threshold_margin(
            top_score,
            relevance_threshold,
        )
    )

    cited_quality = sum(
        _normalize_cosine(score)
        for score in cited_scores
    ) / len(cited_scores)

    multi_source_support = min(
        len(cited_hits) / 2.0,
        1.0,
    )

    confidence = (
        0.35 * absolute_similarity
        + 0.35 * threshold_margin
        + 0.20 * cited_quality
        + 0.10 * multi_source_support
    )

    return round(
        min(
            max(confidence, 0.0),
            1.0,
        ),
        4,
    )


def _normalize_cosine(
    score: float,
) -> float:
    """Map cosine-style similarity from [-1, 1] to [0, 1]."""

    clipped = min(
        max(score, -1.0),
        1.0,
    )

    return (
        clipped + 1.0
    ) / 2.0


def _normalize_threshold_margin(
    score: float,
    threshold: float,
) -> float:
    """Normalize evidence margin above the calibrated gate."""

    if threshold >= 1.0:
        return (
            1.0
            if score >= 1.0
            else 0.0
        )

    normalized = (
        score - threshold
    ) / (
        1.0 - threshold
    )

    return min(
        max(normalized, 0.0),
        1.0,
    )


def _validate_threshold(
    threshold: float,
) -> None:
    if not isfinite(threshold):
        raise EvidenceConfidenceError(
            "Relevance threshold must be finite."
        )

    if not -1.0 <= threshold <= 1.0:
        raise EvidenceConfidenceError(
            "Relevance threshold must be between -1.0 and 1.0."
        )


def _validate_score(
    score: float,
    *,
    name: str,
) -> None:
    if not isfinite(score):
        raise EvidenceConfidenceError(
            f"{name} must be finite."
        )

    if not -1.0 <= score <= 1.0:
        raise EvidenceConfidenceError(
            f"{name} must be between -1.0 and 1.0."
        )