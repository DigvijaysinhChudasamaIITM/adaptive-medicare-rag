from __future__ import annotations

from typing import Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerLike(Protocol):
    """Minimal tokenizer interface required by the RAG pipeline."""

    model_max_length: int

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]: ...


def load_tokenizer(
    model_name: str,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer associated with the embedding model."""
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(
    text: str,
    tokenizer: TokenizerLike,
) -> int:
    """Count model tokens without truncating input text."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
    )