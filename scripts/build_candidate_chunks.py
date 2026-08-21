from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from app.config import get_settings
from app.rag.chunking import (
    ChunkTargetStatistics,
    build_chunks,
    build_structural_sections,
    derive_candidate_targets,
)
from app.rag.pdf_parser import parse_pdf
from app.rag.tokenization import load_tokenizer

PDF_PATH = Path("data/medicare.pdf")
TOKEN_PROFILE_PATH = Path(
    "artifacts/token_profile.json"
)
OUTPUT_DIR = Path("artifacts/chunks")
SUMMARY_PATH = Path(
    "artifacts/chunk_strategy_profile.json"
)


def load_profile() -> dict:
    """Load the measured token profile."""
    return json.loads(
        TOKEN_PROFILE_PATH.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    """Build and validate all document-derived chunk strategies."""
    settings = get_settings()

    profile = load_profile()

    tokenizer = load_tokenizer(
        settings.embedding_model
    )

    document = parse_pdf(PDF_PATH)

    sections = build_structural_sections(
        document.units
    )

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

    candidate_targets = derive_candidate_targets(
        statistics
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_source_orders = {
        unit.source_order
        for unit in document.units
    }

    strategy_summaries: list[dict[str, object]] = []

    for target_tokens in candidate_targets:
        chunks = build_chunks(
            document_id=document.document_id,
            sections=sections,
            target_tokens=target_tokens,
            tokenizer=tokenizer,
            hard_limit=statistics.model_max_length,
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks produced for target {target_tokens}."
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError(
                f"Duplicate chunk IDs for target {target_tokens}."
            )

        represented_source_orders = {
            source_order
            for chunk in chunks
            for source_order in chunk.source_unit_orders
        }

        if represented_source_orders != expected_source_orders:
            missing = (
                expected_source_orders
                - represented_source_orders
            )

            raise RuntimeError(
                "Source content was lost during chunking. "
                f"Missing unit orders: {sorted(missing)[:10]}"
            )

        maximum_tokens = max(
            chunk.token_count
            for chunk in chunks
        )

        if maximum_tokens > statistics.model_max_length:
            raise RuntimeError(
                f"Target {target_tokens} produced a "
                f"{maximum_tokens}-token chunk."
            )

        output = {
            "document_id": document.document_id,
            "document_sha256": document.document_sha256,
            "strategy_id": f"target_{target_tokens}",
            "target_tokens": target_tokens,
            "hard_limit": statistics.model_max_length,
            "chunk_count": len(chunks),
            "chunks": [
                asdict(chunk)
                for chunk in chunks
            ],
        }

        output_path = (
            OUTPUT_DIR
            / f"target_{target_tokens}.json"
        )

        output_path.write_text(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        token_counts = [
            chunk.token_count
            for chunk in chunks
        ]

        chunks_over_soft_target = sum(
            token_count > target_tokens
            for token_count in token_counts
        )

        strategy_summaries.append(
            {
                "strategy_id": f"target_{target_tokens}",
                "target_tokens": target_tokens,
                "chunk_count": len(chunks),
                "mean_tokens": round(
                    mean(token_counts),
                    2,
                ),
                "median_tokens": median(
                    token_counts
                ),
                "max_tokens": maximum_tokens,
                "chunks_over_soft_target": (
                    chunks_over_soft_target
                ),
            }
        )

        print(
            f"Built target {target_tokens}: "
            f"{len(chunks)} chunks, "
            f"max={maximum_tokens}"
        )

    summary = {
        "document_id": document.document_id,
        "document_sha256": document.document_sha256,
        "embedding_model": settings.embedding_model,
        "hard_limit": statistics.model_max_length,
        "candidate_targets": list(
            candidate_targets
        ),
        "strategies": strategy_summaries,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Strategy profile written to "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()