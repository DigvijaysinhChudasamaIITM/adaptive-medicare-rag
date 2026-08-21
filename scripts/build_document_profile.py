from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")
OUTPUT_PATH = Path("artifacts/document_profile.json")


def main() -> None:
    """Build a reproducible profile of the parsed source document."""
    document = parse_pdf(PDF_PATH)

    unit_counts = Counter(
        unit.unit_type
        for unit in document.units
    )

    pages_with_units = {
        unit.page_number
        for unit in document.units
    }

    pages_without_units = [
        page_number
        for page_number in range(
            1,
            document.page_count + 1,
        )
        if page_number not in pages_with_units
    ]

    profile = {
        "document_id": document.document_id,
        "document_sha256": document.document_sha256,
        "page_count": document.page_count,
        "body_font_size": document.body_font_size,
        "semantic_unit_count": len(document.units),
        "unit_counts": {
            "heading": unit_counts.get("heading", 0),
            "paragraph": unit_counts.get("paragraph", 0),
            "list_item": unit_counts.get("list_item", 0),
        },
        "pages_without_semantic_units": pages_without_units,
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

    print(f"Document profile written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()