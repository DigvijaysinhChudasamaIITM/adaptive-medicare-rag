import json
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import pytest

from app.rag.manifest import (
    ArtifactCompatibilityError,
    build_manifest,
    file_sha256,
    validate_runtime_compatibility,
)

DOCUMENT_ID = "medicare"

DOCUMENT_HASH_PLACEHOLDER = "replaced-after-pdf-write"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

STRATEGY_ID = "target_416"

TARGET_TOKENS = 416

EMBEDDING_DIMENSION = 4

CHUNK_COUNT = 2

THRESHOLD = 0.76


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def create_pdf(
    path: Path,
) -> None:
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Medicare test document",
    )

    document.save(
        path
    )

    document.close()


def create_fixture_artifacts(
    tmp_path: Path,
) -> dict[str, Path]:
    pdf_path = (
        tmp_path
        / "medicare.pdf"
    )

    create_pdf(
        pdf_path
    )

    document_sha256 = (
        file_sha256(
            pdf_path
        )
    )

    selected_path = (
        tmp_path
        / "selected_strategy.json"
    )

    relevance_path = (
        tmp_path
        / "relevance_calibration.json"
    )

    index_directory = (
        tmp_path
        / "selected"
    )

    index_directory.mkdir()

    chunks = [
        {
            "chunk_id": "chunk-1",
            "strategy_id": (
                STRATEGY_ID
            ),
            "target_tokens": (
                TARGET_TOKENS
            ),
        },
        {
            "chunk_id": "chunk-2",
            "strategy_id": (
                STRATEGY_ID
            ),
            "target_tokens": (
                TARGET_TOKENS
            ),
        },
    ]

    selected = {
        "document_id": (
            DOCUMENT_ID
        ),
        "document_sha256": (
            document_sha256
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "strategy_id": (
            STRATEGY_ID
        ),
        "target_tokens": (
            TARGET_TOKENS
        ),
        "chunk_count": (
            CHUNK_COUNT
        ),
    }

    calibration = {
        "document_id": (
            DOCUMENT_ID
        ),
        "document_sha256": (
            document_sha256
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "strategy_id": (
            STRATEGY_ID
        ),
        "target_tokens": (
            TARGET_TOKENS
        ),
        "selected_threshold": (
            THRESHOLD
        ),
        "score_definition": (
            "rank_1 normalized embedding "
            "inner-product similarity"
        ),
    }

    metadata = {
        "document_id": (
            DOCUMENT_ID
        ),
        "document_sha256": (
            document_sha256
        ),
        "embedding_model": (
            EMBEDDING_MODEL
        ),
        "embedding_dimension": (
            EMBEDDING_DIMENSION
        ),
        "strategy_id": (
            STRATEGY_ID
        ),
        "target_tokens": (
            TARGET_TOKENS
        ),
        "chunk_count": (
            CHUNK_COUNT
        ),
        "chunks": chunks,
    }

    write_json(
        selected_path,
        selected,
    )

    write_json(
        relevance_path,
        calibration,
    )

    write_json(
        (
            index_directory
            / "metadata.json"
        ),
        metadata,
    )

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    vectors = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float32,
    )

    index.add(
        vectors
    )

    faiss.write_index(
        index,
        str(
            index_directory
            / "index.faiss"
        ),
    )

    manifest_path = (
        tmp_path
        / "manifest.json"
    )

    manifest = build_manifest(
        pdf_path=pdf_path,
        selected_strategy_path=(
            selected_path
        ),
        relevance_calibration_path=(
            relevance_path
        ),
        index_directory=(
            index_directory
        ),
    )

    write_json(
        manifest_path,
        manifest,
    )

    return {
        "pdf": pdf_path,
        "selected": (
            selected_path
        ),
        "relevance": (
            relevance_path
        ),
        "index_directory": (
            index_directory
        ),
        "manifest": (
            manifest_path
        ),
    }


def validate(
    paths: dict[str, Path],
    *,
    embedding_model: str = (
        EMBEDDING_MODEL
    ),
):
    return (
        validate_runtime_compatibility(
            manifest_path=(
                paths[
                    "manifest"
                ]
            ),
            pdf_path=(
                paths[
                    "pdf"
                ]
            ),
            selected_strategy_path=(
                paths[
                    "selected"
                ]
            ),
            relevance_calibration_path=(
                paths[
                    "relevance"
                ]
            ),
            index_directory=(
                paths[
                    "index_directory"
                ]
            ),
            expected_embedding_model=(
                embedding_model
            ),
        )
    )


def test_file_sha256_is_deterministic(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "example.bin"
    )

    path.write_bytes(
        b"medicare"
    )

    first = file_sha256(
        path
    )

    second = file_sha256(
        path
    )

    assert first == second

    assert len(
        first
    ) == 64


def test_runtime_compatibility_accepts_matching_artifacts(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    result = validate(
        paths
    )

    assert (
        result.document_sha256
        == file_sha256(
            paths[
                "pdf"
            ]
        )
    )

    assert (
        result.embedding_model
        == EMBEDDING_MODEL
    )

    assert (
        result.embedding_dimension
        == EMBEDDING_DIMENSION
    )

    assert (
        result.strategy_id
        == STRATEGY_ID
    )

    assert (
        result.target_tokens
        == TARGET_TOKENS
    )

    assert (
        result.chunk_count
        == CHUNK_COUNT
    )

    assert (
        result.index_type
        == "IndexFlatIP"
    )

    assert (
        result.relevance_threshold
        == pytest.approx(
            THRESHOLD
        )
    )


def test_runtime_compatibility_rejects_modified_pdf(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    with paths[
        "pdf"
    ].open(
        "ab"
    ) as handle:
        handle.write(
            b"modified"
        )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="PDF SHA-256",
    ):
        validate(
            paths
        )


def test_runtime_compatibility_rejects_embedding_model_mismatch(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match=(
            "configured embedding model"
        ),
    ):
        validate(
            paths,
            embedding_model=(
                "different/model"
            ),
        )


def test_runtime_compatibility_rejects_strategy_mismatch(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    calibration = json.loads(
        paths[
            "relevance"
        ].read_text(
            encoding="utf-8"
        )
    )

    calibration[
        "strategy_id"
    ] = "target_192"

    write_json(
        paths[
            "relevance"
        ],
        calibration,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="strategy_id",
    ):
        validate(
            paths
        )


def test_runtime_compatibility_rejects_chunk_count_mismatch(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    metadata_path = (
        paths[
            "index_directory"
        ]
        / "metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata[
        "chunk_count"
    ] = 3

    write_json(
        metadata_path,
        metadata,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="chunk count",
    ):
        validate(
            paths
        )


def test_runtime_compatibility_rejects_manifest_tampering(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    manifest = json.loads(
        paths[
            "manifest"
        ].read_text(
            encoding="utf-8"
        )
    )

    manifest[
        "chunking"
    ][
        "target_tokens"
    ] = 192

    write_json(
        paths[
            "manifest"
        ],
        manifest,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="manifest target tokens",
    ):
        validate(
            paths
        )


def test_runtime_compatibility_rejects_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    metadata_path = (
        paths[
            "index_directory"
        ]
        / "metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata[
        "chunks"
    ][1][
        "chunk_id"
    ] = metadata[
        "chunks"
    ][0][
        "chunk_id"
    ]

    write_json(
        metadata_path,
        metadata,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="Duplicate persisted chunk ID",
    ):
        validate(
            paths
        )

def test_runtime_compatibility_rejects_metadata_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    metadata_path = (
        paths[
            "index_directory"
        ]
        / "metadata.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata[
        "chunks"
    ][0][
        "extra_test_field"
    ] = "tampered"

    write_json(
        metadata_path,
        metadata,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="metadata SHA-256",
    ):
        validate(
            paths
        )

def test_runtime_compatibility_rejects_index_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    paths = (
        create_fixture_artifacts(
            tmp_path
        )
    )

    replacement = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    replacement.add(
        np.array(
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float32,
        )
    )

    faiss.write_index(
        replacement,
        str(
            paths[
                "index_directory"
            ]
            / "index.faiss"
        ),
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="FAISS index SHA-256",
    ):
        validate(
            paths
        )