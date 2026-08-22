from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import (
    FaissChunkIndex,
    SearchHit,
)


class QueryEmbedderLike(Protocol):
    """Minimal query-embedding interface used by retrieval."""

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Embed one search query."""


class ChunkIndexLike(Protocol):
    """Minimal vector-index interface used by retrieval."""

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> tuple[SearchHit, ...]:
        """Return ranked search hits."""


@dataclass(slots=True)
class Retriever:
    """Retrieve relevant document chunks for a user query."""

    embedder: QueryEmbedderLike
    index: ChunkIndexLike
    default_top_k: int = 10

    def __post_init__(self) -> None:
        if self.default_top_k <= 0:
            raise ValueError(
                "default_top_k must be positive."
            )

    @classmethod
    def load(
        cls,
        *,
        index_directory: Path,
        embedding_model: str,
        default_top_k: int = 10,
    ) -> Retriever:
        """Load the embedding model and persisted FAISS index."""
        embedder = EmbeddingService.load(
            embedding_model
        )

        index = FaissChunkIndex.load(
            index_directory
        )

        return cls(
            embedder=embedder,
            index=index,
            default_top_k=default_top_k,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[SearchHit, ...]:
        """Retrieve the highest-ranking chunks for a query."""
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "Query must not be empty."
            )

        requested_top_k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        query_embedding = (
            self.embedder.embed_query(
                cleaned_query
            )
        )

        return self.index.search(
            query_embedding,
            top_k=requested_top_k,
        )