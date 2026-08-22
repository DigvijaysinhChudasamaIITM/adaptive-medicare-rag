import numpy as np
import pytest

from app.models.document import DocumentChunk
from app.rag.retrieval import Retriever
from app.rag.vector_store import SearchHit


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        strategy_id="target_192",
        target_tokens=192,
        token_count=30,
        text="Medicare example content.",
        heading_context=("Medicare",),
        page_numbers=(10,),
        page_start=10,
        page_end=10,
        section_index=1,
        chunk_index=0,
        source_unit_orders=(1,),
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        self.queries.append(query)

        return np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )


class FakeIndex:
    def __init__(self) -> None:
        self.top_k_values: list[int] = []

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> tuple[SearchHit, ...]:
        self.top_k_values.append(top_k)

        return (
            SearchHit(
                rank=1,
                score=0.9,
                chunk=make_chunk(),
            ),
        )


def test_retriever_uses_default_top_k() -> None:
    embedder = FakeEmbedder()
    index = FakeIndex()

    retriever = Retriever(
        embedder=embedder,
        index=index,
        default_top_k=10,
    )

    hits = retriever.retrieve(
        "Medicare eligibility"
    )

    assert len(hits) == 1
    assert embedder.queries == [
        "Medicare eligibility"
    ]
    assert index.top_k_values == [10]


def test_retriever_allows_top_k_override() -> None:
    index = FakeIndex()

    retriever = Retriever(
        embedder=FakeEmbedder(),
        index=index,
        default_top_k=10,
    )

    retriever.retrieve(
        "Medicare eligibility",
        top_k=4,
    )

    assert index.top_k_values == [4]


def test_retriever_strips_query_whitespace() -> None:
    embedder = FakeEmbedder()

    retriever = Retriever(
        embedder=embedder,
        index=FakeIndex(),
    )

    retriever.retrieve(
        "  Medicare Part B  "
    )

    assert embedder.queries == [
        "Medicare Part B"
    ]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n",
    ],
)
def test_retriever_rejects_empty_query(
    query: str,
) -> None:
    retriever = Retriever(
        embedder=FakeEmbedder(),
        index=FakeIndex(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(query)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_retriever_rejects_invalid_top_k(
    top_k: int,
) -> None:
    retriever = Retriever(
        embedder=FakeEmbedder(),
        index=FakeIndex(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "Medicare",
            top_k=top_k,
        )


def test_retriever_rejects_invalid_default_top_k() -> None:
    with pytest.raises(ValueError):
        Retriever(
            embedder=FakeEmbedder(),
            index=FakeIndex(),
            default_top_k=0,
        )