from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.models.document import DocumentChunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked chunk returned from vector search."""

    rank: int
    score: float
    chunk: DocumentChunk


class FaissChunkIndex:
    """Exact cosine-style retrieval over normalized embeddings."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[DocumentChunk],
    ) -> None:
        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS vector count does not match "
                "chunk metadata count."
            )

        self.index = index
        self.chunks = chunks

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ) -> FaissChunkIndex:
        """Build an exact inner-product FAISS index."""
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        if matrix.shape[0] != len(chunks):
            raise ValueError(
                "Embedding row count must match "
                "the number of chunks."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Cannot build an empty vector index."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Embeddings contain non-finite values."
            )

        matrix = np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )

        dimension = matrix.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(matrix)

        return cls(
            index=index,
            chunks=chunks,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> tuple[SearchHit, ...]:
        """Return the highest-scoring chunks."""
        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if (
            query.ndim != 2
            or query.shape[0] != 1
        ):
            raise ValueError(
                "Query embedding must contain "
                "exactly one vector."
            )

        if query.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does "
                "not match the FAISS index."
            )

        query = np.ascontiguousarray(
            query,
            dtype=np.float32,
        )

        result_count = min(
            top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query,
            result_count,
        )

        hits: list[SearchHit] = []

        for rank, (
            score,
            row_index,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
                strict=True,
            ),
            start=1,
        ):
            if row_index < 0:
                continue

            hits.append(
                SearchHit(
                    rank=rank,
                    score=float(score),
                    chunk=self.chunks[
                        int(row_index)
                    ],
                )
            )

        return tuple(hits)

    def save(
        self,
        directory: Path,
        *,
        document_id: str,
        document_sha256: str,
        embedding_model: str,
        strategy_id: str,
        target_tokens: int,
    ) -> None:
        """Persist FAISS index and matching chunk metadata."""
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            directory / "index.faiss"
        )

        metadata_path = (
            directory / "metadata.json"
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata = {
            "document_id": document_id,
            "document_sha256": document_sha256,
            "embedding_model": embedding_model,
            "embedding_dimension": self.index.d,
            "strategy_id": strategy_id,
            "target_tokens": target_tokens,
            "chunk_count": len(self.chunks),
            "chunks": [
                asdict(chunk)
                for chunk in self.chunks
            ],
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> FaissChunkIndex:
        """Load a persisted FAISS index and its metadata."""
        index_path = (
            directory / "index.faiss"
        )

        metadata_path = (
            directory / "metadata.json"
        )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Index metadata not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        chunks = [
            _chunk_from_dict(item)
            for item in metadata["chunks"]
        ]

        return cls(
            index=index,
            chunks=chunks,
        )


def _chunk_from_dict(
    item: dict[str, Any],
) -> DocumentChunk:
    """Reconstruct a DocumentChunk from JSON metadata."""
    return DocumentChunk(
        chunk_id=str(item["chunk_id"]),
        strategy_id=str(
            item["strategy_id"]
        ),
        target_tokens=int(
            item["target_tokens"]
        ),
        token_count=int(
            item["token_count"]
        ),
        text=str(item["text"]),
        heading_context=tuple(
            str(value)
            for value in item[
                "heading_context"
            ]
        ),
        page_numbers=tuple(
            int(value)
            for value in item[
                "page_numbers"
            ]
        ),
        page_start=int(
            item["page_start"]
        ),
        page_end=int(
            item["page_end"]
        ),
        section_index=int(
            item["section_index"]
        ),
        chunk_index=int(
            item["chunk_index"]
        ),
        source_unit_orders=tuple(
            int(value)
            for value in item[
                "source_unit_orders"
            ]
        ),
    )