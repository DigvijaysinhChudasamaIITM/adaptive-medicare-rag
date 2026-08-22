from __future__ import annotations

import json
import shutil
from pathlib import Path

EVALUATION_PATH = Path(
    "artifacts/retrieval_evaluation.json"
)

INDEX_ROOT = Path(
    "artifacts/indexes"
)

OUTPUT_PATH = Path(
    "artifacts/selected_strategy.json"
)

PRODUCTION_INDEX_DIR = Path(
    "artifacts/indexes/selected"
)


def main() -> None:
    """Persist the empirically selected chunk strategy."""
    evaluation = json.loads(
        EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    selected = evaluation[
        "selected_strategy"
    ]

    strategy_id = str(
        selected["strategy_id"]
    )

    target_tokens = int(
        selected["target_tokens"]
    )

    strategies = {
        str(strategy["strategy_id"]): strategy
        for strategy in evaluation[
            "strategies"
        ]
    }

    if strategy_id not in strategies:
        raise RuntimeError(
            "Selected strategy is missing "
            "from evaluation results."
        )

    strategy_result = strategies[
        strategy_id
    ]

    source_index_dir = (
        INDEX_ROOT / strategy_id
    )

    if not source_index_dir.exists():
        raise FileNotFoundError(
            f"Selected index not found: "
            f"{source_index_dir}"
        )

    if PRODUCTION_INDEX_DIR.exists():
        shutil.rmtree(
            PRODUCTION_INDEX_DIR
        )

    shutil.copytree(
        source_index_dir,
        PRODUCTION_INDEX_DIR,
    )

    output = {
        "document_id": evaluation[
            "document_id"
        ],
        "document_sha256": evaluation[
            "document_sha256"
        ],
        "embedding_model": evaluation[
            "embedding_model"
        ],
        "strategy_id": strategy_id,
        "target_tokens": target_tokens,
        "selection_policy": evaluation[
            "selection_policy"
        ],
        "aggregate_metrics": (
            strategy_result[
                "aggregate"
            ]
        ),
        "mean_chunk_tokens": (
            strategy_result[
                "mean_chunk_tokens"
            ]
        ),
        "chunk_count": (
            strategy_result[
                "chunk_count"
            ]
        ),
        "production_index_directory": (
            str(PRODUCTION_INDEX_DIR)
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

    print(
        f"Selected strategy: "
        f"{strategy_id}"
    )

    print(
        f"Target tokens: "
        f"{target_tokens}"
    )

    print(
        f"Production index copied to: "
        f"{PRODUCTION_INDEX_DIR}"
    )

    print(
        f"Selection artifact written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()