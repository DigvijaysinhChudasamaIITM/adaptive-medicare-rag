from pathlib import Path

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")
PAGES_TO_INSPECT = (10, 11, 17, 80, 118, 127)


def main() -> None:
    """Print reconstructed semantic units for representative pages."""
    document = parse_pdf(PDF_PATH)

    print("PARSED DOCUMENT SUMMARY")
    print("=" * 100)
    print(f"Document ID: {document.document_id}")
    print(f"SHA-256: {document.document_sha256}")
    print(f"Physical pages: {document.page_count}")
    print(f"Body font size: {document.body_font_size}")
    print(f"Semantic units: {len(document.units)}")

    for page_number in PAGES_TO_INSPECT:
        page_units = [
            unit
            for unit in document.units
            if unit.page_number == page_number
        ]

        print()
        print("=" * 100)
        print(
            f"PHYSICAL PAGE {page_number} "
            f"— {len(page_units)} SEMANTIC UNITS"
        )
        print("=" * 100)

        if not page_units:
            print("<NO SEMANTIC UNITS>")
            continue

        for unit in page_units:
            print()
            print(
                f"[{unit.source_order:04d}] "
                f"type={unit.unit_type:<10} "
                f"font={unit.font_size:>4.1f} "
                f"x={unit.bbox.x0:>6.1f} "
                f"y={unit.bbox.y0:>6.1f}"
            )
            print(unit.text)


if __name__ == "__main__":
    main()