from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from pathlib import Path

import pymupdf

PDF_PATH = Path("data/medicare.pdf")

# Physical PDF pages, 1-based.
#
# Selected to represent:
# - cover/content pages
# - multi-column comparison layouts
# - Medicare enrollment content
# - Part D content
# - definitions/end matter
SAMPLE_PAGES = [1, 3, 10, 11, 15, 17, 80,118, 119, 127, 128]


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_character_counts(document: pymupdf.Document) -> list[int]:
    """Return extracted text character counts for every physical PDF page."""
    return [
        len(page.get_text("text", sort=True).strip())
        for page in document
    ]


def get_font_size_distribution(
    document: pymupdf.Document,
) -> Counter[float]:
    """Measure how much text appears at each font size."""
    sizes: Counter[float] = Counter()

    for page in document:
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    font_size = round(float(span["size"]), 1)

                    # Weight by character count instead of span count.
                    sizes[font_size] += len(text)

    return sizes


def print_page_blocks(
    page: pymupdf.Page,
    limit: int = 15,
) -> None:
    """Print positional information for the first text blocks on a page."""
    blocks = page.get_text("blocks", sort=True)

    for index, block in enumerate(blocks[:limit], start=1):
        x0, y0, x1, y1, text, *_ = block

        clean_text = " ".join(text.split())

        print(
            f"{index:02d}. "
            f"x0={x0:7.1f} "
            f"y0={y0:7.1f} "
            f"x1={x1:7.1f} "
            f"y1={y1:7.1f} | "
            f"{clean_text[:180]}"
        )


def print_page_sample(
    document: pymupdf.Document,
    page_number: int,
) -> None:
    """Inspect one physical PDF page in multiple extraction modes."""
    page = document[page_number - 1]

    unsorted_text = page.get_text("text")
    sorted_text = page.get_text("text", sort=True)

    blocks = page.get_text("blocks", sort=True)

    print()
    print("=" * 100)
    print(f"PHYSICAL PDF PAGE: {page_number}")
    print("=" * 100)

    print(f"Width: {page.rect.width:.1f}")
    print(f"Height: {page.rect.height:.1f}")
    print(f"Sorted character count: {len(sorted_text):,}")
    print(f"Text block count: {len(blocks):,}")

    print()
    print("--- SORTED TEXT SAMPLE ---")
    print(sorted_text[:1800].strip())

    print()
    print("--- UNSORTED TEXT SAMPLE ---")
    print(unsorted_text[:1200].strip())

    print()
    print("--- FIRST POSITIONED TEXT BLOCKS ---")
    print_page_blocks(page)


def main() -> None:
    """Inspect the Medicare PDF before implementing production ingestion."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Expected assignment PDF was not found at: {PDF_PATH}"
        )

    print("MEDICARE PDF INSPECTION")
    print("=" * 100)

    print(f"Path: {PDF_PATH}")
    print(f"File size: {PDF_PATH.stat().st_size:,} bytes")
    print(f"SHA-256: {calculate_sha256(PDF_PATH)}")

    with pymupdf.open(PDF_PATH) as document:
        print(f"Page count: {document.page_count}")

        character_counts = get_character_counts(document)

        print()
        print("TEXT DISTRIBUTION")
        print("-" * 100)

        print(
            f"Minimum characters/page: "
            f"{min(character_counts):,}"
        )
        print(
            f"Median characters/page: "
            f"{statistics.median(character_counts):,.0f}"
        )
        print(
            f"Maximum characters/page: "
            f"{max(character_counts):,}"
        )
        print(
            f"Total extracted characters: "
            f"{sum(character_counts):,}"
        )

        low_text_pages = [
            page_number
            for page_number, character_count
            in enumerate(character_counts, start=1)
            if character_count < 100
        ]

        print(
            "Pages with <100 extracted characters: "
            f"{low_text_pages or 'None'}"
        )

        font_sizes = get_font_size_distribution(document)

        print()
        print("MOST COMMON FONT SIZES")
        print("-" * 100)

        for font_size, character_count in font_sizes.most_common(12):
            print(
                f"{font_size:>5.1f} pt -> "
                f"{character_count:,} characters"
            )

        for page_number in SAMPLE_PAGES:
            if page_number <= document.page_count:
                print_page_sample(
                    document=document,
                    page_number=page_number,
                )


if __name__ == "__main__":
    main()