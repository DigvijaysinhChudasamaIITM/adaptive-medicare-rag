import numpy as np
import pytest

from app.rag.embeddings import (
    BGE_QUERY_INSTRUCTION,
    EmbeddingService,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(sentences)

        return np.array(
            [
                [1.0, 0.0, 0.0]
                for _ in sentences
            ],
            dtype=np.float32,
        )


def test_document_embedding_does_not_add_query_instruction() -> None:
    encoder = FakeEncoder()

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        model=encoder,
    )

    embeddings = service.embed_documents(
        ["Medicare Part B coverage"]
    )

    assert embeddings.shape == (1, 3)

    assert encoder.calls[-1] == [
        "Medicare Part B coverage"
    ]


def test_bge_query_embedding_adds_retrieval_instruction() -> None:
    encoder = FakeEncoder()

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        model=encoder,
    )

    embedding = service.embed_query(
        "What does Part B cover?"
    )

    assert embedding.shape == (3,)

    assert encoder.calls[-1] == [
        BGE_QUERY_INSTRUCTION
        + "What does Part B cover?"
    ]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
    ],
)
def test_query_embedding_rejects_empty_query(
    query: str,
) -> None:
    service = EmbeddingService(
        model_name="test-model",
        model=FakeEncoder(),
    )

    with pytest.raises(ValueError):
        service.embed_query(query)