from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.rag.embeddings import EmbeddingService
from app.rag.evaluation import (
    group_recall_at_k,
    load_golden_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.rag.vector_store import FaissChunkIndex

GOLD_PATH = Path(
    "evaluation/golden_queries.json"
)

INDEX_PROFILE_PATH = Path(
    "artifacts/index_profile.json"
)

INDEX_ROOT = Path(
    "artifacts/indexes"
)

OUTPUT_PATH = Path(
    "artifacts/retrieval_evaluation.json"
)

K_VALUES = (
    1,
    3,
    5,
)


def load_index_profile() -> dict:
    """Load candidate index metadata."""
    return json.loads(
        INDEX_PROFILE_PATH.read_text(
            encoding="utf-8"
        )
    )


def average(
    values: list[float],
) -> float:
    """Return a stable mean for non-empty metric lists."""
    if not values:
        return 0.0

    return float(mean(values))


def main() -> None:
    """Evaluate every candidate chunk strategy."""
    settings = get_settings()

    golden_queries = (
        load_golden_queries(
            GOLD_PATH
        )
    )

    profile = load_index_profile()

    embedder = EmbeddingService.load(
        settings.embedding_model
    )

    print(
        f"Embedding {len(golden_queries)} "
        "evaluation queries..."
    )

    query_embeddings = {
        query.query_id: (
            embedder.embed_query(
                query.query
            )
        )
        for query in golden_queries
    }

    strategy_results: list[
        dict[str, object]
    ] = []

    for strategy in profile[
        "strategies"
    ]:
        strategy_id = str(
            strategy["strategy_id"]
        )

        target_tokens = int(
            strategy["target_tokens"]
        )

        print()
        print(
            f"Evaluating {strategy_id}..."
        )

        index = FaissChunkIndex.load(
            INDEX_ROOT / strategy_id
        )

        per_query: list[
            dict[str, object]
        ] = []

        for query in golden_queries:
            hits = index.search(
                query_embeddings[
                    query.query_id
                ],
                top_k=max(K_VALUES),
            )

            metrics: dict[
                str,
                float,
            ] = {}

            for k in K_VALUES:
                metrics[
                    f"precision_at_{k}"
                ] = precision_at_k(
                    hits,
                    query,
                    k,
                )

                metrics[
                    f"recall_at_{k}"
                ] = recall_at_k(
                    hits,
                    query,
                    k,
                )

            metrics[
                "group_recall_at_5"
            ] = group_recall_at_k(
                hits,
                query,
                5,
            )

            metrics[
                "mrr_at_5"
            ] = reciprocal_rank_at_k(
                hits,
                query,
                5,
            )

            metrics[
                "ndcg_at_5"
            ] = ndcg_at_k(
                hits,
                query,
                index.chunks,
                5,
            )

            per_query.append(
                {
                    "query_id": (
                        query.query_id
                    ),
                    "category": (
                        query.category
                    ),
                    "metrics": metrics,
                    "top_hits": [
                        {
                            "rank": hit.rank,
                            "score": round(
                                hit.score,
                                6,
                            ),
                            "chunk_id": (
                                hit.chunk.chunk_id
                            ),
                            "pages": list(
                                hit.chunk.page_numbers
                            ),
                        }
                        for hit in hits
                    ],
                }
            )

        metric_names = (
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "group_recall_at_5",
            "mrr_at_5",
            "ndcg_at_5",
        )

        aggregate = {
            metric_name: average(
                [
                    float(
                        result[
                            "metrics"
                        ][metric_name]
                    )
                    for result in per_query
                ]
            )
            for metric_name in metric_names
        }

        mean_chunk_tokens = average(
            [
                float(
                    chunk.token_count
                )
                for chunk in index.chunks
            ]
        )

        strategy_results.append(
            {
                "strategy_id": (
                    strategy_id
                ),
                "target_tokens": (
                    target_tokens
                ),
                "chunk_count": len(
                    index.chunks
                ),
                "mean_chunk_tokens": (
                    mean_chunk_tokens
                ),
                "aggregate": (
                    aggregate
                ),
                "per_query": (
                    per_query
                ),
            }
        )

    selected = max(
        strategy_results,
        key=selection_key,
    )

    output = {
        "document_id": profile[
            "document_id"
        ],
        "document_sha256": profile[
            "document_sha256"
        ],
        "embedding_model": profile[
            "embedding_model"
        ],
        "query_count": len(
            golden_queries
        ),
        "k_values": list(
            K_VALUES
        ),
        "selection_policy": [
            "highest mean Recall@5",
            "then highest mean MRR@5",
            "then highest mean NDCG@5",
            "then highest mean Precision@5",
            "then lower mean chunk-token count",
        ],
        "strategies": strategy_results,
        "selected_strategy": {
            "strategy_id": selected[
                "strategy_id"
            ],
            "target_tokens": selected[
                "target_tokens"
            ],
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print_summary(
        strategy_results
    )

    print()
    print(
        "SELECTED:",
        selected["strategy_id"],
        f"(target={selected['target_tokens']})",
    )

    print()
    print(
        f"Evaluation report written to "
        f"{OUTPUT_PATH}"
    )


def selection_key(
    result: dict[str, object],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """Return deterministic strategy-selection ranking."""
    aggregate = result["aggregate"]

    if not isinstance(
        aggregate,
        dict,
    ):
        raise TypeError(
            "aggregate metrics must be a dictionary."
        )

    return (
        float(
            aggregate["recall_at_5"]
        ),
        float(
            aggregate["mrr_at_5"]
        ),
        float(
            aggregate["ndcg_at_5"]
        ),
        float(
            aggregate["precision_at_5"]
        ),
        -float(
            result[
                "mean_chunk_tokens"
            ]
        ),
    )


def print_summary(
    results: list[
        dict[str, object]
    ],
) -> None:
    """Print compact strategy comparison."""
    print(
        f"{'strategy':<12}"
        f"{'P@5':>9}"
        f"{'R@5':>9}"
        f"{'GR@5':>9}"
        f"{'MRR@5':>9}"
        f"{'NDCG@5':>10}"
        f"{'mean tok':>11}"
    )

    print(
        "-" * 78
    )

    for result in results:
        aggregate = result[
            "aggregate"
        ]

        if not isinstance(
            aggregate,
            dict,
        ):
            raise TypeError(
                "aggregate metrics must be a dictionary."
            )

        print(
            f"{str(result['strategy_id']):<12}"
            f"{float(aggregate['precision_at_5']):>9.4f}"
            f"{float(aggregate['recall_at_5']):>9.4f}"
            f"{float(aggregate['group_recall_at_5']):>9.4f}"
            f"{float(aggregate['mrr_at_5']):>9.4f}"
            f"{float(aggregate['ndcg_at_5']):>10.4f}"
            f"{float(result['mean_chunk_tokens']):>11.1f}"
        )


if __name__ == "__main__":
    main()