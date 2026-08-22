from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.rag.evaluation import (
    load_golden_queries,
    load_negative_queries,
    select_relevance_threshold,
)
from app.rag.retrieval import Retriever
from app.rag.vector_store import SearchHit

GOLD_PATH = Path(
    "evaluation/golden_queries.json"
)

NEGATIVE_PATH = Path(
    "evaluation/negative_queries.json"
)

SELECTED_STRATEGY_PATH = Path(
    "artifacts/selected_strategy.json"
)

SELECTED_INDEX_PATH = Path(
    "artifacts/indexes/selected"
)

OUTPUT_PATH = Path(
    "artifacts/relevance_calibration.json"
)


def retrieve_top_hit(
    retriever: Retriever,
    query: str,
) -> SearchHit:
    """Retrieve exactly one top-ranked chunk."""
    hits = retriever.retrieve(
        query,
        top_k=1,
    )

    if not hits:
        raise RuntimeError(
            "Expected the selected index "
            "to return at least one hit."
        )

    return hits[0]


def score_summary(
    scores: list[float],
) -> dict[str, float]:
    """Return compact descriptive statistics for retrieval scores."""
    return {
        "min": min(
            scores
        ),
        "mean": float(
            mean(
                scores
            )
        ),
        "max": max(
            scores
        ),
    }


def main() -> None:
    """Calibrate no-answer gating from positive and negative queries."""
    settings = get_settings()

    positives = (
        load_golden_queries(
            GOLD_PATH
        )
    )

    negatives = (
        load_negative_queries(
            NEGATIVE_PATH
        )
    )

    selected_strategy = json.loads(
        SELECTED_STRATEGY_PATH.read_text(
            encoding="utf-8"
        )
    )

    retriever = Retriever.load(
        index_directory=(
            SELECTED_INDEX_PATH
        ),
        embedding_model=(
            settings.embedding_model
        ),
        default_top_k=1,
    )

    positive_rows: list[
        dict[str, object]
    ] = []

    negative_rows: list[
        dict[str, object]
    ] = []

    print(
        "Scoring positive queries..."
    )

    for query in positives:
        hit = retrieve_top_hit(
            retriever,
            query.query,
        )

        positive_rows.append(
            {
                "query_id": (
                    query.query_id
                ),
                "category": (
                    query.category
                ),
                "query": query.query,
                "top_score": (
                    float(
                        hit.score
                    )
                ),
                "top_chunk_id": (
                    hit.chunk.chunk_id
                ),
                "page_numbers": list(
                    hit.chunk.page_numbers
                ),
            }
        )

    print(
        "Scoring negative queries..."
    )

    for query in negatives:
        hit = retrieve_top_hit(
            retriever,
            query.query,
        )

        negative_rows.append(
            {
                "query_id": (
                    query.query_id
                ),
                "category": (
                    query.category
                ),
                "query": query.query,
                "top_score": (
                    float(
                        hit.score
                    )
                ),
                "top_chunk_id": (
                    hit.chunk.chunk_id
                ),
                "page_numbers": list(
                    hit.chunk.page_numbers
                ),
            }
        )

    positive_scores = [
        float(
            row["top_score"]
        )
        for row in positive_rows
    ]

    negative_scores = [
        float(
            row["top_score"]
        )
        for row in negative_rows
    ]

    selection = (
        select_relevance_threshold(
            positive_scores,
            negative_scores,
        )
    )

    selected = (
        selection.selected
    )

    for row in positive_rows:
        row["predicted_answerable"] = (
            float(
                row["top_score"]
            )
            >= selected.threshold
        )

    for row in negative_rows:
        row["predicted_answerable"] = (
            float(
                row["top_score"]
            )
            >= selected.threshold
        )

    output = {
        "version": 1,
        "document_id": (
            selected_strategy[
                "document_id"
            ]
        ),
        "document_sha256": (
            selected_strategy[
                "document_sha256"
            ]
        ),
        "embedding_model": (
            selected_strategy[
                "embedding_model"
            ]
        ),
        "strategy_id": (
            selected_strategy[
                "strategy_id"
            ]
        ),
        "target_tokens": (
            selected_strategy[
                "target_tokens"
            ]
        ),
        "score_definition": (
            "rank_1 normalized embedding "
            "inner-product similarity"
        ),
        "positive_query_count": len(
            positive_rows
        ),
        "negative_query_count": len(
            negative_rows
        ),
        "threshold_selection_policy": [
            "highest balanced accuracy",
            "then highest positive recall",
            "then highest negative specificity",
            "then lower threshold",
        ],
        "candidates_evaluated": (
            selection.candidates_evaluated
        ),
        "selected_threshold": (
            selected.threshold
        ),
        "metrics": asdict(
            selected
        ),
        "positive_score_summary": (
            score_summary(
                positive_scores
            )
        ),
        "negative_score_summary": (
            score_summary(
                negative_scores
            )
        ),
        "positive_queries": (
            positive_rows
        ),
        "negative_queries": (
            negative_rows
        ),
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
    print(
        "Positive queries"
    )

    print(
        "-" * 72
    )

    for row in positive_rows:
        print(
            f"{str(row['query_id']):<5}"
            f"{float(row['top_score']):>10.4f}  "
            f"pages={row['page_numbers']}  "
            f"{row['category']}"
        )

    print()
    print(
        "Negative queries"
    )

    print(
        "-" * 72
    )

    for row in negative_rows:
        print(
            f"{str(row['query_id']):<5}"
            f"{float(row['top_score']):>10.4f}  "
            f"pages={row['page_numbers']}  "
            f"{row['category']}"
        )

    positive_summary = (
        output[
            "positive_score_summary"
        ]
    )

    negative_summary = (
        output[
            "negative_score_summary"
        ]
    )

    if not isinstance(
        positive_summary,
        dict,
    ):
        raise TypeError(
            "Positive score summary must be a dictionary."
        )

    if not isinstance(
        negative_summary,
        dict,
    ):
        raise TypeError(
            "Negative score summary must be a dictionary."
        )

    print()
    print(
        "Calibration summary"
    )

    print(
        "-" * 72
    )

    print(
        "positive min:       "
        f"{float(positive_summary['min']):.4f}"
    )

    print(
        "positive mean:      "
        f"{float(positive_summary['mean']):.4f}"
    )

    print(
        "positive max:       "
        f"{float(positive_summary['max']):.4f}"
    )

    print(
        "negative min:       "
        f"{float(negative_summary['min']):.4f}"
    )

    print(
        "negative mean:      "
        f"{float(negative_summary['mean']):.4f}"
    )

    print(
        "negative max:       "
        f"{float(negative_summary['max']):.4f}"
    )

    print()
    print(
        "selected threshold: "
        f"{selected.threshold:.6f}"
    )

    print(
        "positive recall:    "
        f"{selected.positive_recall:.4f}"
    )

    print(
        "negative specificity: "
        f"{selected.negative_specificity:.4f}"
    )

    print(
        "balanced accuracy:  "
        f"{selected.balanced_accuracy:.4f}"
    )

    print(
        "candidates tested:  "
        f"{selection.candidates_evaluated}"
    )

    print()
    print(
        "Calibration artifact written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()