from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from math import isfinite, log2, nextafter
from pathlib import Path

from app.models.document import DocumentChunk
from app.models.evaluation import (
    GoldenQuery,
    NegativeQuery,
    ThresholdMetrics,
    ThresholdSelection,
)
from app.rag.vector_store import SearchHit


def load_golden_queries(
    path: Path,
) -> tuple[GoldenQuery, ...]:
    """Load and validate manually labeled retrieval queries."""
    if not path.exists():
        raise FileNotFoundError(
            f"Golden query file not found: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    raw_queries = payload.get("queries")

    if not isinstance(raw_queries, list):
        raise ValueError(
            "Golden query file must contain a queries list."
        )

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()

    for item in raw_queries:
        query_id = str(item["query_id"]).strip()
        query = str(item["query"]).strip()
        category = str(item["category"]).strip()

        if not query_id:
            raise ValueError(
                "query_id must not be empty."
            )

        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate query_id: {query_id}"
            )

        if not query:
            raise ValueError(
                f"Query {query_id} must not be empty."
            )

        raw_groups = item["evidence_groups"]

        if not raw_groups:
            raise ValueError(
                f"Query {query_id} must contain evidence groups."
            )

        evidence_groups = tuple(
            tuple(
                int(source_order)
                for source_order in group
            )
            for group in raw_groups
        )

        if any(
            not group
            for group in evidence_groups
        ):
            raise ValueError(
                f"Query {query_id} contains an empty evidence group."
            )

        gold_pages = tuple(
            int(page)
            for page in item["gold_pages"]
        )

        queries.append(
            GoldenQuery(
                query_id=query_id,
                query=query,
                category=category,
                evidence_groups=evidence_groups,
                gold_pages=gold_pages,
            )
        )

        seen_ids.add(query_id)

    if not queries:
        raise ValueError(
            "Golden query dataset must not be empty."
        )

    return tuple(queries)

def load_negative_queries(
    path: Path,
) -> tuple[NegativeQuery, ...]:
    """Load and validate no-answer threshold calibration queries."""
    if not path.exists():
        raise FileNotFoundError(
            f"Negative query file not found: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    raw_queries = payload.get(
        "queries"
    )

    if not isinstance(
        raw_queries,
        list,
    ):
        raise ValueError(
            "Negative query file must contain a queries list."
        )

    queries: list[
        NegativeQuery
    ] = []

    seen_ids: set[str] = set()

    for item in raw_queries:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Every negative query entry must be an object."
            )

        query_id = str(
            item["query_id"]
        ).strip()

        query = str(
            item["query"]
        ).strip()

        category = str(
            item["category"]
        ).strip()

        if not query_id:
            raise ValueError(
                "query_id must not be empty."
            )

        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate query_id: {query_id}"
            )

        if not query:
            raise ValueError(
                f"Query {query_id} must not be empty."
            )

        if not category:
            raise ValueError(
                f"Query {query_id} must contain a category."
            )

        queries.append(
            NegativeQuery(
                query_id=query_id,
                query=query,
                category=category,
            )
        )

        seen_ids.add(
            query_id
        )

    if not queries:
        raise ValueError(
            "Negative query dataset must not be empty."
        )

    return tuple(
        queries
    )

def gold_source_units(
    query: GoldenQuery,
) -> frozenset[int]:
    """Return all unique gold source units for one query."""
    return frozenset(
        source_order
        for group in query.evidence_groups
        for source_order in group
    )


def retrieved_source_units(
    hits: Sequence[SearchHit],
    k: int,
) -> frozenset[int]:
    """Return unique source units represented in the top-k hits."""
    _validate_k(k)

    return frozenset(
        source_order
        for hit in hits[:k]
        for source_order in hit.chunk.source_unit_orders
    )


def is_relevant_hit(
    hit: SearchHit,
    query: GoldenQuery,
) -> bool:
    """Return whether a hit contains any gold evidence."""
    gold_units = gold_source_units(query)

    return bool(
        gold_units.intersection(
            hit.chunk.source_unit_orders
        )
    )


def precision_at_k(
    hits: Sequence[SearchHit],
    query: GoldenQuery,
    k: int,
) -> float:
    """Calculate chunk-level Precision@K."""
    _validate_k(k)

    relevant_count = sum(
        1
        for hit in hits[:k]
        if is_relevant_hit(hit, query)
    )

    return relevant_count / k


def recall_at_k(
    hits: Sequence[SearchHit],
    query: GoldenQuery,
    k: int,
) -> float:
    """Calculate source-evidence Recall@K."""
    _validate_k(k)

    gold_units = gold_source_units(query)

    if not gold_units:
        return 0.0

    retrieved_units = retrieved_source_units(
        hits,
        k,
    )

    covered_units = (
        gold_units.intersection(
            retrieved_units
        )
    )

    return (
        len(covered_units)
        / len(gold_units)
    )


def group_recall_at_k(
    hits: Sequence[SearchHit],
    query: GoldenQuery,
    k: int,
) -> float:
    """Calculate equal-weighted evidence-group coverage."""
    _validate_k(k)

    retrieved_units = retrieved_source_units(
        hits,
        k,
    )

    group_coverages: list[float] = []

    for group in query.evidence_groups:
        group_units = set(group)

        covered = group_units.intersection(
            retrieved_units
        )

        group_coverages.append(
            len(covered)
            / len(group_units)
        )

    if not group_coverages:
        return 0.0

    return (
        sum(group_coverages)
        / len(group_coverages)
    )


def reciprocal_rank_at_k(
    hits: Sequence[SearchHit],
    query: GoldenQuery,
    k: int,
) -> float:
    """Calculate reciprocal rank of the first relevant hit."""
    _validate_k(k)

    for rank, hit in enumerate(
        hits[:k],
        start=1,
    ):
        if is_relevant_hit(hit, query):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    hits: Sequence[SearchHit],
    query: GoldenQuery,
    corpus_chunks: Sequence[DocumentChunk],
    k: int,
) -> float:
    """Calculate binary relevance NDCG@K."""
    _validate_k(k)

    relevances = [
        1.0
        if is_relevant_hit(hit, query)
        else 0.0
        for hit in hits[:k]
    ]

    dcg = _dcg(relevances)

    gold_units = gold_source_units(query)

    relevant_chunk_count = sum(
        1
        for chunk in corpus_chunks
        if gold_units.intersection(
            chunk.source_unit_orders
        )
    )

    ideal_relevant_count = min(
        k,
        relevant_chunk_count,
    )

    if ideal_relevant_count == 0:
        return 0.0

    ideal_relevances = [
        1.0
        for _ in range(
            ideal_relevant_count
        )
    ]

    idcg = _dcg(
        ideal_relevances
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def _dcg(
    relevances: Sequence[float],
) -> float:
    """Calculate discounted cumulative gain."""
    return sum(
        relevance
        / log2(rank + 1)
        for rank, relevance in enumerate(
            relevances,
            start=1,
        )
    )


def _validate_k(k: int) -> None:
    """Validate retrieval metric cutoff."""
    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

def evaluate_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate one relevance threshold as a binary classifier."""
    positives = _validated_scores(
        positive_scores,
        label="positive",
    )

    negatives = _validated_scores(
        negative_scores,
        label="negative",
    )

    if not isfinite(
        threshold
    ):
        raise ValueError(
            "Threshold must be finite."
        )

    true_positive = sum(
        score >= threshold
        for score in positives
    )

    false_negative = (
        len(positives)
        - true_positive
    )

    true_negative = sum(
        score < threshold
        for score in negatives
    )

    false_positive = (
        len(negatives)
        - true_negative
    )

    positive_recall = (
        true_positive
        / len(positives)
    )

    negative_specificity = (
        true_negative
        / len(negatives)
    )

    balanced_accuracy = (
        positive_recall
        + negative_specificity
    ) / 2.0

    return ThresholdMetrics(
        threshold=float(
            threshold
        ),
        true_positive=(
            true_positive
        ),
        false_negative=(
            false_negative
        ),
        true_negative=(
            true_negative
        ),
        false_positive=(
            false_positive
        ),
        positive_recall=(
            positive_recall
        ),
        negative_specificity=(
            negative_specificity
        ),
        balanced_accuracy=(
            balanced_accuracy
        ),
    )


def select_relevance_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> ThresholdSelection:
    """Select a threshold from measured positive and negative scores."""
    positives = _validated_scores(
        positive_scores,
        label="positive",
    )

    negatives = _validated_scores(
        negative_scores,
        label="negative",
    )

    candidates = (
        _candidate_thresholds(
            (
                *positives,
                *negatives,
            )
        )
    )

    metrics = tuple(
        evaluate_threshold(
            positives,
            negatives,
            threshold,
        )
        for threshold in candidates
    )

    selected = max(
        metrics,
        key=lambda item: (
            item.balanced_accuracy,
            item.positive_recall,
            item.negative_specificity,
            -item.threshold,
        ),
    )

    return ThresholdSelection(
        selected=selected,
        candidates_evaluated=(
            len(metrics)
        ),
    )


def _candidate_thresholds(
    scores: Sequence[float],
) -> tuple[float, ...]:
    """Create deterministic thresholds from observed score boundaries."""
    values = sorted(
        set(
            float(score)
            for score in scores
        )
    )

    if not values:
        raise ValueError(
            "At least one score is required."
        )

    candidates: list[
        float
    ] = [
        values[0]
    ]

    for left, right in pairwise(
        values
    ):
        candidates.append(
            (
                left
                + right
            )
            / 2.0
        )

    candidates.append(
        nextafter(
            values[-1],
            float("inf"),
        )
    )

    return tuple(
        candidates
    )


def _validated_scores(
    scores: Sequence[float],
    *,
    label: str,
) -> tuple[float, ...]:
    """Validate a non-empty collection of retrieval scores."""
    values = tuple(
        float(score)
        for score in scores
    )

    if not values:
        raise ValueError(
            f"{label.capitalize()} scores must not be empty."
        )

    if any(
        not isfinite(score)
        for score in values
    ):
        raise ValueError(
            f"{label.capitalize()} scores must be finite."
        )

    return values
