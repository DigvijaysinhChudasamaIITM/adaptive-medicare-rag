from __future__ import annotations

import json
from collections.abc import Sequence
from math import log2
from pathlib import Path

from app.models.document import DocumentChunk
from app.models.evaluation import GoldenQuery
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