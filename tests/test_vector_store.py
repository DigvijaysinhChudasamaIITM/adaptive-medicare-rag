import numpy as np

from app.models.document import DocumentChunk
from app.rag.vector_store import (
    FaissChunkIndex,
)


def make_chunk(
    chunk_id: str,
    chunk_index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        strategy_id="target_128",
        target_tokens=128,
        token_count=20,
        text=f"Text for {chunk_id}",
        heading_context=("Test heading",),
        page_numbers=(1,),
        page_start=1,
        page_end=1,
        section_index=0,
        chunk_index=chunk_index,
        source_unit_orders=(chunk_index,),
    )


def test_faiss_search_returns_best_match_first() -> None:
    chunks = [
        make_chunk("chunk-a", 0),
        make_chunk("chunk-b", 1),
        make_chunk("chunk-c", 2),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store = FaissChunkIndex.build(
        embeddings,
        chunks,
    )

    hits = store.search(
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=2,
    )

    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "chunk-a"
    assert hits[0].rank == 1
    assert hits[0].score > hits[1].score


def test_faiss_search_caps_top_k_to_chunk_count() -> None:
    chunks = [
        make_chunk("chunk-a", 0),
        make_chunk("chunk-b", 1),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = FaissChunkIndex.build(
        embeddings,
        chunks,
    )

    hits = store.search(
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=10,
    )

    assert len(hits) == 2


def test_faiss_index_round_trip(
    tmp_path,
) -> None:
    chunks = [
        make_chunk("chunk-a", 0),
        make_chunk("chunk-b", 1),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = FaissChunkIndex.build(
        embeddings,
        chunks,
    )

    directory = (
        tmp_path / "test-index"
    )

    store.save(
        directory,
        document_id="test",
        document_sha256="abc123",
        embedding_model="test-model",
        strategy_id="target_128",
        target_tokens=128,
    )

    loaded = FaissChunkIndex.load(
        directory
    )

    hits = loaded.search(
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "chunk-a"