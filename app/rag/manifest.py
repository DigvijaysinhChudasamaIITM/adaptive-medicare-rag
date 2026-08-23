from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import pymupdf

MANIFEST_VERSION = 1

EXPECTED_INDEX_TYPE = "IndexFlatIP"


class ArtifactCompatibilityError(RuntimeError):
    """Raised when persisted RAG artifacts are mutually incompatible."""


@dataclass(
    frozen=True,
    slots=True,
)
class CompatibilityResult:
    """Successful compatibility validation summary."""

    document_sha256: str
    embedding_model: str
    embedding_dimension: int
    strategy_id: str
    target_tokens: int
    chunk_count: int
    index_type: str
    relevance_threshold: float


def file_sha256(
    path: Path,
) -> str:
    """Calculate SHA-256 for a file without loading it all into memory."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON artifact and require an object at the root."""
    if not path.exists():
        raise FileNotFoundError(
            f"JSON artifact not found: {path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"JSON artifact must contain an object: {path}"
        )

    return payload


def build_manifest(
    *,
    pdf_path: Path,
    selected_strategy_path: Path,
    relevance_calibration_path: Path,
    index_directory: Path,
) -> dict[str, Any]:
    """Build a manifest from the current persisted retrieval artifacts."""
    selected = load_json_object(
        selected_strategy_path
    )

    calibration = load_json_object(
        relevance_calibration_path
    )

    metadata_path = (
        index_directory
        / "metadata.json"
    )

    index_path = (
        index_directory
        / "index.faiss"
    )

    metadata = load_json_object(
        metadata_path
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    index_sha256 = file_sha256(
        index_path
    )

    metadata_sha256 = file_sha256(
        metadata_path
    )

    index = faiss.read_index(
        str(
            index_path
        )
    )

    pdf_hash = file_sha256(
        pdf_path
    )

    with pymupdf.open(
        pdf_path
    ) as document:
        page_count = (
            document.page_count
        )

    _validate_cross_artifact_values(
        pdf_sha256=pdf_hash,
        selected=selected,
        calibration=calibration,
        metadata=metadata,
        index=index,
        expected_embedding_model=None,
    )

    return {
        "manifest_version": (
            MANIFEST_VERSION
        ),
        "document": {
            "document_id": str(
                selected[
                    "document_id"
                ]
            ),
            "path": (
                pdf_path.as_posix()
            ),
            "sha256": pdf_hash,
            "page_count": int(
                page_count
            ),
            "byte_size": int(
                pdf_path.stat().st_size
            ),
        },
        "embedding": {
            "model": str(
                selected[
                    "embedding_model"
                ]
            ),
            "dimension": int(
                metadata[
                    "embedding_dimension"
                ]
            ),
        },
        "chunking": {
            "strategy_id": str(
                selected[
                    "strategy_id"
                ]
            ),
            "target_tokens": int(
                selected[
                    "target_tokens"
                ]
            ),
            "chunk_count": int(
                selected[
                    "chunk_count"
                ]
            ),
        },
        "index": {
            "type": (
                type(index).__name__
            ),
            "dimension": int(
                index.d
            ),
            "vector_count": int(
                index.ntotal
            ),
            "directory": (
                index_directory.as_posix()
            ),
            "index_sha256": (
                index_sha256
            ),
            "metadata_sha256": (
                metadata_sha256
            ),
        },
        "relevance": {
            "threshold": float(
                calibration[
                    "selected_threshold"
                ]
            ),
            "score_definition": str(
                calibration[
                    "score_definition"
                ]
            ),
        },
        "source_artifacts": {
            "selected_strategy": (
                selected_strategy_path.as_posix()
            ),
            "relevance_calibration": (
                relevance_calibration_path.as_posix()
            ),
            "index_metadata": (
                metadata_path.as_posix()
            ),
        },
    }

def validate_runtime_compatibility(
    *,
    manifest_path: Path,
    pdf_path: Path,
    selected_strategy_path: Path,
    relevance_calibration_path: Path,
    index_directory: Path,
    expected_embedding_model: str,
) -> CompatibilityResult:
    """Validate that all persisted retrieval artifacts belong together."""
    manifest = load_json_object(
        manifest_path
    )

    selected = load_json_object(
        selected_strategy_path
    )

    calibration = load_json_object(
        relevance_calibration_path
    )

    metadata_path = (
        index_directory
        / "metadata.json"
    )

    index_path = (
        index_directory
        / "index.faiss"
    )

    metadata = load_json_object(
        metadata_path
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    index = faiss.read_index(
        str(
            index_path
        )
    )

    pdf_hash = file_sha256(
        pdf_path
    )

    _validate_cross_artifact_values(
        pdf_sha256=pdf_hash,
        selected=selected,
        calibration=calibration,
        metadata=metadata,
        index=index,
        expected_embedding_model=(
            expected_embedding_model
        ),
    )

    _validate_manifest_values(
        manifest=manifest,
        pdf_path=pdf_path,
        pdf_sha256=pdf_hash,
        selected=selected,
        calibration=calibration,
        metadata=metadata,
        index=index,
        index_directory=index_directory,
    )

    return CompatibilityResult(
        document_sha256=pdf_hash,
        embedding_model=str(
            selected[
                "embedding_model"
            ]
        ),
        embedding_dimension=int(
            metadata[
                "embedding_dimension"
            ]
        ),
        strategy_id=str(
            selected[
                "strategy_id"
            ]
        ),
        target_tokens=int(
            selected[
                "target_tokens"
            ]
        ),
        chunk_count=int(
            selected[
                "chunk_count"
            ]
        ),
        index_type=(
            type(index).__name__
        ),
        relevance_threshold=float(
            calibration[
                "selected_threshold"
            ]
        ),
    )


def _validate_cross_artifact_values(
    *,
    pdf_sha256: str,
    selected: dict[str, Any],
    calibration: dict[str, Any],
    metadata: dict[str, Any],
    index: Any,
    expected_embedding_model: str | None,
) -> None:
    """Validate compatibility without relying on the manifest itself."""
    _require_equal(
        "PDF SHA-256 vs selected strategy",
        pdf_sha256,
        selected[
            "document_sha256"
        ],
    )

    _require_equal(
        "PDF SHA-256 vs index metadata",
        pdf_sha256,
        metadata[
            "document_sha256"
        ],
    )

    _require_equal(
        "PDF SHA-256 vs relevance calibration",
        pdf_sha256,
        calibration[
            "document_sha256"
        ],
    )

    for field in (
        "document_id",
        "embedding_model",
        "strategy_id",
        "target_tokens",
    ):
        _require_equal(
            (
                "selected strategy vs "
                f"index metadata: {field}"
            ),
            selected[field],
            metadata[field],
        )

        _require_equal(
            (
                "selected strategy vs "
                f"relevance calibration: {field}"
            ),
            selected[field],
            calibration[field],
        )

    if (
        expected_embedding_model
        is not None
    ):
        _require_equal(
            (
                "configured embedding model "
                "vs persisted embedding model"
            ),
            expected_embedding_model,
            selected[
                "embedding_model"
            ],
        )

    selected_chunk_count = int(
        selected[
            "chunk_count"
        ]
    )

    metadata_chunk_count = int(
        metadata[
            "chunk_count"
        ]
    )

    _require_equal(
        "selected strategy vs index chunk count",
        selected_chunk_count,
        metadata_chunk_count,
    )

    chunks = metadata.get(
        "chunks"
    )

    if not isinstance(
        chunks,
        list,
    ):
        raise ArtifactCompatibilityError(
            "Index metadata must contain a chunks list."
        )

    _require_equal(
        "index metadata chunk list length",
        metadata_chunk_count,
        len(
            chunks
        ),
    )

    _require_equal(
        "FAISS vector count vs metadata chunk count",
        int(
            index.ntotal
        ),
        metadata_chunk_count,
    )

    embedding_dimension = int(
        metadata[
            "embedding_dimension"
        ]
    )

    _require_equal(
        "FAISS dimension vs metadata embedding dimension",
        int(
            index.d
        ),
        embedding_dimension,
    )

    index_type = (
        type(index).__name__
    )

    _require_equal(
        "FAISS index type",
        index_type,
        EXPECTED_INDEX_TYPE,
    )

    chunk_ids: set[str] = set()

    expected_strategy_id = str(
        selected[
            "strategy_id"
        ]
    )

    expected_target_tokens = int(
        selected[
            "target_tokens"
        ]
    )

    for chunk in chunks:
        if not isinstance(
            chunk,
            dict,
        ):
            raise ArtifactCompatibilityError(
                "Every persisted chunk metadata entry "
                "must be an object."
            )

        chunk_id = str(
            chunk[
                "chunk_id"
            ]
        )

        if chunk_id in chunk_ids:
            raise ArtifactCompatibilityError(
                "Duplicate persisted chunk ID: "
                f"{chunk_id}"
            )

        chunk_ids.add(
            chunk_id
        )

        _require_equal(
            (
                f"chunk {chunk_id} strategy"
            ),
            str(
                chunk[
                    "strategy_id"
                ]
            ),
            expected_strategy_id,
        )

        _require_equal(
            (
                f"chunk {chunk_id} target tokens"
            ),
            int(
                chunk[
                    "target_tokens"
                ]
            ),
            expected_target_tokens,
        )

def _validate_manifest_values(
    *,
    manifest: dict[str, Any],
    pdf_path: Path,
    pdf_sha256: str,
    selected: dict[str, Any],
    calibration: dict[str, Any],
    metadata: dict[str, Any],
    index: Any,
    index_directory: Path,
) -> None:
    """Validate the persisted manifest against live artifacts."""
    _require_equal(
        "manifest version",
        manifest[
            "manifest_version"
        ],
        MANIFEST_VERSION,
    )

    document = manifest[
        "document"
    ]

    embedding = manifest[
        "embedding"
    ]

    chunking = manifest[
        "chunking"
    ]

    index_manifest = manifest[
        "index"
    ]

    relevance = manifest[
        "relevance"
    ]

    index_path = (
        index_directory
        / "index.faiss"
    )

    metadata_path = (
        index_directory
        / "metadata.json"
    )

    _require_equal(
        "manifest document ID",
        document[
            "document_id"
        ],
        selected[
            "document_id"
        ],
    )

    _require_equal(
        "manifest document path",
        document[
            "path"
        ],
        pdf_path.as_posix(),
    )

    _require_equal(
        "manifest PDF SHA-256",
        document[
            "sha256"
        ],
        pdf_sha256,
    )

    _require_equal(
        "manifest PDF byte size",
        int(
            document[
                "byte_size"
            ]
        ),
        int(
            pdf_path.stat().st_size
        ),
    )

    with pymupdf.open(
        pdf_path
    ) as pdf_document:
        page_count = (
            pdf_document.page_count
        )

    _require_equal(
        "manifest PDF page count",
        int(
            document[
                "page_count"
            ]
        ),
        int(
            page_count
        ),
    )

    _require_equal(
        "manifest embedding model",
        embedding[
            "model"
        ],
        selected[
            "embedding_model"
        ],
    )

    _require_equal(
        "manifest embedding dimension",
        int(
            embedding[
                "dimension"
            ]
        ),
        int(
            metadata[
                "embedding_dimension"
            ]
        ),
    )

    _require_equal(
        "manifest chunking strategy",
        chunking[
            "strategy_id"
        ],
        selected[
            "strategy_id"
        ],
    )

    _require_equal(
        "manifest target tokens",
        int(
            chunking[
                "target_tokens"
            ]
        ),
        int(
            selected[
                "target_tokens"
            ]
        ),
    )

    _require_equal(
        "manifest chunk count",
        int(
            chunking[
                "chunk_count"
            ]
        ),
        int(
            selected[
                "chunk_count"
            ]
        ),
    )

    _require_equal(
        "manifest index type",
        index_manifest[
            "type"
        ],
        type(index).__name__,
    )

    _require_equal(
        "manifest index dimension",
        int(
            index_manifest[
                "dimension"
            ]
        ),
        int(
            index.d
        ),
    )

    _require_equal(
        "manifest index vector count",
        int(
            index_manifest[
                "vector_count"
            ]
        ),
        int(
            index.ntotal
        ),
    )

    _require_equal(
        "manifest index directory",
        index_manifest[
            "directory"
        ],
        index_directory.as_posix(),
    )

    _require_equal(
        "manifest FAISS index SHA-256",
        index_manifest[
            "index_sha256"
        ],
        file_sha256(
            index_path
        ),
    )

    _require_equal(
        "manifest index metadata SHA-256",
        index_manifest[
            "metadata_sha256"
        ],
        file_sha256(
            metadata_path
        ),
    )

    _require_equal(
        "manifest relevance threshold",
        float(
            relevance[
                "threshold"
            ]
        ),
        float(
            calibration[
                "selected_threshold"
            ]
        ),
    )

    _require_equal(
        "manifest relevance score definition",
        relevance[
            "score_definition"
        ],
        calibration[
            "score_definition"
        ],
    )


def _require_equal(
    label: str,
    actual: object,
    expected: object,
) -> None:
    """Raise a descriptive compatibility error for mismatched values."""
    if actual != expected:
        raise ArtifactCompatibilityError(
            f"{label} mismatch: "
            f"{actual!r} != {expected!r}"
        )