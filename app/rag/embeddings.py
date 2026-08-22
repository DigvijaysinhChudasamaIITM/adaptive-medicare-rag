from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EncoderModelLike(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode input texts."""


@dataclass(slots=True)
class EmbeddingService:
    """Encode document chunks and search queries."""

    model_name: str
    model: EncoderModelLike
    batch_size: int = 32

    @classmethod
    def load(
        cls,
        model_name: str,
        batch_size: int = 32,
    ) -> EmbeddingService:
        """Load a sentence-transformer embedding model."""
        if batch_size <= 0:
            raise ValueError(
                "Embedding batch size must be positive."
            )

        model = SentenceTransformer(model_name)

        return cls(
            model_name=model_name,
            model=model,
            batch_size=batch_size,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Create normalized document embeddings."""
        if not texts:
            raise ValueError(
                "At least one document text is required."
            )

        cleaned_texts = [
            text.strip()
            for text in texts
        ]

        if any(not text for text in cleaned_texts):
            raise ValueError(
                "Document texts must not be empty."
            )

        return self._encode(cleaned_texts)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Create a normalized query embedding."""
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "Query must not be empty."
            )

        query_text = self._prepare_query(
            cleaned_query
        )

        embeddings = self._encode([query_text])

        return embeddings[0]

    def _prepare_query(
        self,
        query: str,
    ) -> str:
        """Apply model-specific retrieval instructions."""
        if self.model_name.lower() == (
            "baai/bge-small-en-v1.5"
        ):
            return (
                BGE_QUERY_INSTRUCTION
                + query
            )

        return query

    def _encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Encode texts into contiguous float32 vectors."""
        raw_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            raw_embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(
                1,
                -1,
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding model returned an invalid shape."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embedding model returned non-finite values."
            )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )