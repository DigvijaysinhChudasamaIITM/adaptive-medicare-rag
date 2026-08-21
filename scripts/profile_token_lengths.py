from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from app.config import get_settings
from app.rag.chunking import build_structural_sections, section_text
from app.rag.pdf_parser import parse_pdf
from app.rag.tokenization import count_tokens, load_tokenizer

PDF_PATH = Path("data/medicare.pdf")
OUTPUT_PATH = Path("artifacts/token_profile.json")


def summarize_lengths(
    lengths: list[int],
) -> dict[str, float | int]:
    """Return descriptive statistics for a collection of token lengths."""
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0,
        }

    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": round(mean(lengths), 2),
        "median": round(median(lengths), 2),
        "p25": round(float(np.percentile(lengths, 25)), 2),
        "p75": round(float(np.percentile(lengths, 75)), 2),
        "p90": round(float(np.percentile(lengths, 90)), 2),
        "p95": round(float(np.percentile(lengths, 95)), 2),
        "max": max(lengths),
    }

def main() -> None:
    """Profile semantic-unit and structural-section token lengths."""
    settings = get_settings()

    print(f"Loading tokenizer: {settings.embedding_model}")

    tokenizer = load_tokenizer(
        settings.embedding_model
    )

    document = parse_pdf(PDF_PATH)

    lengths_by_type: dict[str, list[int]] = defaultdict(list)
    all_unit_lengths: list[int] = []

    oversized_units: list[dict[str, object]] = []

    for unit in document.units:
        token_count = count_tokens(
            unit.text,
            tokenizer,
        )

        all_unit_lengths.append(token_count)

        lengths_by_type[unit.unit_type].append(
            token_count
        )

        if token_count > tokenizer.model_max_length:
            oversized_units.append(
                {
                    "page_number": unit.page_number,
                    "unit_type": unit.unit_type,
                    "source_order": unit.source_order,
                    "token_count": token_count,
                    "text_preview": unit.text[:160],
                }
            )

    sections = build_structural_sections(
        document.units
    )

    section_lengths: list[int] = []
    oversized_sections: list[dict[str, object]] = []

    for section_index, section in enumerate(sections):
        text = section_text(section)

        token_count = count_tokens(
            text,
            tokenizer,
        )

        section_lengths.append(token_count)

        if token_count > tokenizer.model_max_length:
            oversized_sections.append(
                {
                    "section_index": section_index,
                    "page_start": min(
                        unit.page_number
                        for unit in section
                    ),
                    "page_end": max(
                        unit.page_number
                        for unit in section
                    ),
                    "unit_count": len(section),
                    "token_count": token_count,
                    "heading_preview": next(
                        (
                            unit.text[:160]
                            for unit in section
                            if unit.unit_type == "heading"
                        ),
                        None,
                    ),
                }
            )

    profile = {
        "document_id": document.document_id,
        "document_sha256": document.document_sha256,
        "embedding_model": settings.embedding_model,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "unit_statistics": {
            "all": summarize_lengths(
                all_unit_lengths
            ),
            "heading": summarize_lengths(
                lengths_by_type["heading"]
            ),
            "paragraph": summarize_lengths(
                lengths_by_type["paragraph"]
            ),
            "list_item": summarize_lengths(
                lengths_by_type["list_item"]
            ),
        },
        "structural_section_statistics": summarize_lengths(
            section_lengths
        ),
        "structural_section_count": len(sections),
        "oversized_section_count": len(
            oversized_sections
        ),
        "oversized_sections": oversized_sections,
        "oversized_unit_count": len(
            oversized_units
        ),
        "oversized_units": oversized_units,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            profile,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("TOKEN PROFILE")
    print("=" * 72)
    print(
        json.dumps(
            profile,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"Token profile written to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()