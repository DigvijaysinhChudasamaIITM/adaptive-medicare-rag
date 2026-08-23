from __future__ import annotations

import json
from pathlib import Path

from app.rag.manifest import (
    build_manifest,
    validate_runtime_compatibility,
)

PDF_PATH = Path(
    "data/medicare.pdf"
)

SELECTED_STRATEGY_PATH = Path(
    "artifacts/selected_strategy.json"
)

RELEVANCE_CALIBRATION_PATH = Path(
    "artifacts/relevance_calibration.json"
)

SELECTED_INDEX_DIRECTORY = Path(
    "artifacts/indexes/selected"
)

MANIFEST_PATH = Path(
    "artifacts/manifest.json"
)


def main() -> None:
    """Build and immediately validate the production retrieval manifest."""
    manifest = build_manifest(
        pdf_path=PDF_PATH,
        selected_strategy_path=(
            SELECTED_STRATEGY_PATH
        ),
        relevance_calibration_path=(
            RELEVANCE_CALIBRATION_PATH
        ),
        index_directory=(
            SELECTED_INDEX_DIRECTORY
        ),
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = (
        validate_runtime_compatibility(
            manifest_path=(
                MANIFEST_PATH
            ),
            pdf_path=(
                PDF_PATH
            ),
            selected_strategy_path=(
                SELECTED_STRATEGY_PATH
            ),
            relevance_calibration_path=(
                RELEVANCE_CALIBRATION_PATH
            ),
            index_directory=(
                SELECTED_INDEX_DIRECTORY
            ),
            expected_embedding_model=(
                manifest[
                    "embedding"
                ][
                    "model"
                ]
            ),
        )
    )

    print(
        "Manifest written:",
        MANIFEST_PATH,
    )

    print(
        "Compatibility validation: PASS"
    )

    print(
        "document_sha256:",
        result.document_sha256,
    )

    print(
        "embedding_model:",
        result.embedding_model,
    )

    print(
        "embedding_dimension:",
        result.embedding_dimension,
    )

    print(
        "strategy_id:",
        result.strategy_id,
    )

    print(
        "target_tokens:",
        result.target_tokens,
    )

    print(
        "chunk_count:",
        result.chunk_count,
    )

    print(
        "index_type:",
        result.index_type,
    )

    print(
        "relevance_threshold:",
        result.relevance_threshold,
    )


if __name__ == "__main__":
    main()