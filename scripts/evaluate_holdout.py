from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.rag.embeddings import EmbeddingService
from app.rag.evaluation import (
    group_recall_at_k,
    is_relevant_hit,
    load_golden_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.rag.relevance import RelevanceGate
from app.rag.retrieval import Retriever
from app.rag.vector_store import FaissChunkIndex

HOLDOUT_PATH = Path(
    "evaluation/holdout_queries.json"
)

SELECTED_STRATEGY_PATH = Path(
    "artifacts/selected_strategy.json"
)

CALIBRATION_PATH = Path(
    "artifacts/relevance_calibration.json"
)

INDEX_DIRECTORY = Path(
    "artifacts/indexes/selected"
)

OUTPUT_PATH = Path(
    "artifacts/holdout_evaluation.json"
)

TOP_K = 5


def main() -> None:
    """Evaluate the locked production retriever on unseen holdout queries."""
    settings = get_settings()

    holdout_queries = (
        load_golden_queries(
            HOLDOUT_PATH
        )
    )

    selected = json.loads(
        SELECTED_STRATEGY_PATH.read_text(
            encoding="utf-8"
        )
    )

    index = FaissChunkIndex.load(
        INDEX_DIRECTORY
    )

    embedder = EmbeddingService.load(
        settings.embedding_model
    )

    retriever = Retriever(
        embedder=embedder,
        index=index,
        default_top_k=TOP_K,
    )

    relevance_gate = (
        RelevanceGate.load(
            CALIBRATION_PATH
        )
    )

    query_results: list[
        dict[str, object]
    ] = []

    for query in holdout_queries:
        hits = retriever.retrieve(
            query.query,
            top_k=TOP_K,
        )

        decision = (
            relevance_gate.assess(
                hits
            )
        )

        precision_1 = (
            precision_at_k(
                hits,
                query,
                1,
            )
        )

        precision_3 = (
            precision_at_k(
                hits,
                query,
                3,
            )
        )

        precision_5 = (
            precision_at_k(
                hits,
                query,
                5,
            )
        )

        recall_1 = (
            recall_at_k(
                hits,
                query,
                1,
            )
        )

        recall_3 = (
            recall_at_k(
                hits,
                query,
                3,
            )
        )

        recall_5 = (
            recall_at_k(
                hits,
                query,
                5,
            )
        )

        group_recall_5 = (
            group_recall_at_k(
                hits,
                query,
                5,
            )
        )

        reciprocal_rank_5 = (
            reciprocal_rank_at_k(
                hits,
                query,
                5,
            )
        )

        ndcg_5 = (
            ndcg_at_k(
                hits,
                query,
                index.chunks,
                5,
            )
        )

        relevant_ranks = [
            hit.rank
            for hit in hits
            if is_relevant_hit(
                hit,
                query,
            )
        ]

        first_relevant_rank = (
            min(relevant_ranks)
            if relevant_ranks
            else None
        )

        top_score = (
            float(
                hits[0].score
            )
            if hits
            else None
        )

        threshold_margin = (
            top_score
            - relevance_gate.threshold
            if top_score is not None
            else None
        )

        retrieved = [
            {
                "rank": (
                    hit.rank
                ),
                "score": float(
                    hit.score
                ),
                "chunk_id": (
                    hit.chunk.chunk_id
                ),
                "page_numbers": list(
                    hit.chunk.page_numbers
                ),
                "source_unit_orders": list(
                    hit.chunk.source_unit_orders
                ),
                "is_gold_relevant": (
                    is_relevant_hit(
                        hit,
                        query,
                    )
                ),
            }
            for hit in hits
        ]

        row: dict[
            str,
            object,
        ] = {
            "query_id": (
                query.query_id
            ),
            "query": (
                query.query
            ),
            "category": (
                query.category
            ),
            "gold_pages": list(
                query.gold_pages
            ),
            "gold_source_units": sorted(
                {
                    source_order
                    for group
                    in query.evidence_groups
                    for source_order
                    in group
                }
            ),
            "top_score": (
                top_score
            ),
            "relevance_threshold": (
                relevance_gate.threshold
            ),
            "threshold_margin": (
                threshold_margin
            ),
            "accepted_by_relevance_gate": (
                decision.is_relevant
            ),
            "relevance_reason": (
                decision.reason
            ),
            "first_relevant_rank": (
                first_relevant_rank
            ),
            "precision_at_1": (
                precision_1
            ),
            "precision_at_3": (
                precision_3
            ),
            "precision_at_5": (
                precision_5
            ),
            "recall_at_1": (
                recall_1
            ),
            "recall_at_3": (
                recall_3
            ),
            "recall_at_5": (
                recall_5
            ),
            "group_recall_at_5": (
                group_recall_5
            ),
            "mrr_at_5": (
                reciprocal_rank_5
            ),
            "ndcg_at_5": (
                ndcg_5
            ),
            "retrieved_top_5": (
                retrieved
            ),
        }

        query_results.append(
            row
        )

        print()
        print(
            "=" * 100
        )
        print(
            f"{query.query_id}: "
            f"{query.query}"
        )
        print(
            "=" * 100
        )

        print(
            f"top_score: "
            f"{top_score:.6f}"
            if top_score is not None
            else "top_score: None"
        )

        print(
            "gate:",
            decision.is_relevant,
            decision.reason,
        )

        print(
            "threshold_margin:",
            (
                f"{threshold_margin:.6f}"
                if threshold_margin
                is not None
                else "None"
            ),
        )

        print(
            "first_relevant_rank:",
            first_relevant_rank,
        )

        print(
            "Recall@5:",
            f"{recall_5:.4f}",
        )

        print(
            "Group Recall@5:",
            f"{group_recall_5:.4f}",
        )

        print(
            "MRR@5:",
            f"{reciprocal_rank_5:.4f}",
        )

        print(
            "NDCG@5:",
            f"{ndcg_5:.4f}",
        )

        for hit in hits:
            print(
                f"  rank={hit.rank} "
                f"score={hit.score:.4f} "
                f"pages="
                f"{list(hit.chunk.page_numbers)} "
                f"relevant="
                f"{is_relevant_hit(hit, query)} "
                f"chunk="
                f"{hit.chunk.chunk_id}"
            )

    accepted_count = sum(
        1
        for row in query_results
        if row[
            "accepted_by_relevance_gate"
        ]
    )

    aggregate = {
        "precision_at_1": mean(
            float(
                row[
                    "precision_at_1"
                ]
            )
            for row
            in query_results
        ),
        "precision_at_3": mean(
            float(
                row[
                    "precision_at_3"
                ]
            )
            for row
            in query_results
        ),
        "precision_at_5": mean(
            float(
                row[
                    "precision_at_5"
                ]
            )
            for row
            in query_results
        ),
        "recall_at_1": mean(
            float(
                row[
                    "recall_at_1"
                ]
            )
            for row
            in query_results
        ),
        "recall_at_3": mean(
            float(
                row[
                    "recall_at_3"
                ]
            )
            for row
            in query_results
        ),
        "recall_at_5": mean(
            float(
                row[
                    "recall_at_5"
                ]
            )
            for row
            in query_results
        ),
        "group_recall_at_5": mean(
            float(
                row[
                    "group_recall_at_5"
                ]
            )
            for row
            in query_results
        ),
        "mrr_at_5": mean(
            float(
                row[
                    "mrr_at_5"
                ]
            )
            for row
            in query_results
        ),
        "ndcg_at_5": mean(
            float(
                row[
                    "ndcg_at_5"
                ]
            )
            for row
            in query_results
        ),
    }

    artifact = {
        "version": 1,
        "evaluation_type": (
            "independent_holdout_sanity_check"
        ),
        "policy": (
            "The holdout is evaluated only after "
            "chunk-strategy selection and relevance-threshold "
            "calibration were locked. Results are not used "
            "to reselect the strategy or retune the threshold."
        ),
        "document_id": (
            selected[
                "document_id"
            ]
        ),
        "document_sha256": (
            selected[
                "document_sha256"
            ]
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "strategy_id": (
            selected[
                "strategy_id"
            ]
        ),
        "target_tokens": (
            selected[
                "target_tokens"
            ]
        ),
        "query_count": (
            len(
                query_results
            )
        ),
        "top_k": TOP_K,
        "relevance_threshold": (
            relevance_gate.threshold
        ),
        "accepted_query_count": (
            accepted_count
        ),
        "threshold_acceptance_rate": (
            accepted_count
            / len(
                query_results
            )
        ),
        "aggregate_metrics": (
            aggregate
        ),
        "queries": (
            query_results
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )
    print(
        "HOLDOUT SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        "queries:",
        len(
            query_results
        ),
    )

    print(
        "accepted_by_gate:",
        (
            f"{accepted_count}/"
            f"{len(query_results)}"
        ),
    )

    for name, value in aggregate.items():
        print(
            f"{name}: "
            f"{value:.6f}"
        )

    print(
        "artifact:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()