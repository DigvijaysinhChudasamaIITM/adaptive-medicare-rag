from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.rag.chunking import (
    ChunkTargetStatistics,
    build_chunks,
    build_structural_sections,
    compose_chunk_text,
    derive_candidate_targets,
)
from app.rag.embeddings import EmbeddingService
from app.rag.pdf_parser import parse_pdf
from app.rag.tokenization import load_tokenizer
from app.rag.vector_store import FaissChunkIndex

PDF_PATH = Path("data/medicare.pdf")
TOKEN_PROFILE_PATH = Path("artifacts/token_profile.json")
OUTPUT_DIR = Path("artifacts/indexes")
INDEX_PROFILE_PATH = Path("artifacts/index_profile.json")


def load_token_profile() -> dict:
    """Load document token statistics used for candidate derivation."""
    return json.loads(
        TOKEN_PROFILE_PATH.read_text(
            encoding="utf-8"
        )
    )


def derive_targets(
    profile: dict,
) -> tuple[int, ...]:
    """Reconstruct document-derived candidate targets."""
    section_stats = profile[
        "structural_section_statistics"
    ]
    paragraph_stats = profile[
        "unit_statistics"
    ]["paragraph"]

    statistics = ChunkTargetStatistics(
        section_median=float(
            section_stats["median"]
        ),
        section_p75=float(
            section_stats["p75"]
        ),
        section_p90=float(
            section_stats["p90"]
        ),
        section_p95=float(
            section_stats["p95"]
        ),
        paragraph_p95=float(
            paragraph_stats["p95"]
        ),
        model_max_length=int(
            profile["tokenizer_model_max_length"]
        ),
    )

    return derive_candidate_targets(
        statistics
    )


def main() -> None:
    """Build and persist one exact FAISS index per chunk strategy."""
    settings = get_settings()

    profile = load_token_profile()

    tokenizer = load_tokenizer(
        settings.embedding_model
    )

    embedding_service = EmbeddingService.load(
        settings.embedding_model
    )

    document = parse_pdf(PDF_PATH)

    if (
        document.document_sha256
        != profile["document_sha256"]
    ):
        raise RuntimeError(
            "PDF fingerprint does not match the token profile. "
            "Regenerate preprocessing artifacts first."
        )

    sections = build_structural_sections(
        document.units
    )

    candidate_targets = derive_targets(
        profile
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategies: list[dict[str, object]] = []

    for target_tokens in candidate_targets:
        print()
        print(
            f"Building target_{target_tokens}..."
        )

        chunks = build_chunks(
            document_id=document.document_id,
            sections=sections,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            hard_limit=int(
                profile[
                    "tokenizer_model_max_length"
                ]
            ),
        )

        embedding_texts = [
            compose_chunk_text(
                chunk.heading_context,
                chunk.text,
            )
            for chunk in chunks
        ]

        print(
            f"Embedding {len(chunks)} chunks..."
        )

        embeddings = (
            embedding_service.embed_documents(
                embedding_texts
            )
        )

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                "Embedding row count does not match "
                "chunk count."
            )

        store = FaissChunkIndex.build(
            embeddings,
            chunks,
        )

        strategy_id = (
            f"target_{target_tokens}"
        )

        strategy_directory = (
            OUTPUT_DIR / strategy_id
        )

        store.save(
            strategy_directory,
            document_id=document.document_id,
            document_sha256=document.document_sha256,
            embedding_model=settings.embedding_model,
            strategy_id=strategy_id,
            target_tokens=target_tokens,
        )

        strategies.append(
            {
                "strategy_id": strategy_id,
                "target_tokens": target_tokens,
                "chunk_count": len(chunks),
                "embedding_dimension": int(
                    embeddings.shape[1]
                ),
                "index_type": "IndexFlatIP",
            }
        )

        print(
            f"Saved {strategy_id}: "
            f"{len(chunks)} vectors × "
            f"{embeddings.shape[1]} dimensions"
        )

    index_profile = {
        "document_id": document.document_id,
        "document_sha256": document.document_sha256,
        "embedding_model": settings.embedding_model,
        "candidate_targets": list(
            candidate_targets
        ),
        "strategies": strategies,
    }

    INDEX_PROFILE_PATH.write_text(
        json.dumps(
            index_profile,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Index profile written to "
        f"{INDEX_PROFILE_PATH}"
    )


if __name__ == "__main__":
    main()