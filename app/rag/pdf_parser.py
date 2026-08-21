from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pymupdf

from app.models.document import (
    BoundingBox,
    DocumentUnit,
    ParsedDocument,
    TextLine,
    UnitType,
)

_PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,3}$")

_BULLET_PREFIXES = (
    "•",
    "▪",
    "◦",
    "‣",
    "–",
)

_HEADER_MAX_Y = 50.0
_FOOTER_MIN_Y = 735.0
_MIN_REPEATED_BOILERPLATE_PAGES = 3

_CLOSING_PUNCTUATION = frozenset(",.;:!?%)]}")
_NO_SPACE_AFTER = ("(", "[", "{", "$", "£", "€", "/", "-", "–", "—")


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving textual content."""
    return " ".join(text.split())


def normalize_boilerplate_text(text: str) -> str:
    """Normalize text before repeated header/footer comparison."""
    return normalize_whitespace(text).casefold()


def join_spans(spans: list[dict]) -> str:
    """Join PDF spans while preserving visually meaningful word spacing."""
    result = ""
    previous_span: dict | None = None

    for span in spans:
        raw_text = str(span.get("text", ""))
        piece = normalize_whitespace(raw_text)

        if not piece:
            continue

        if previous_span is None:
            result = piece
            previous_span = span
            continue

        previous_raw = str(previous_span.get("text", ""))

        previous_x1 = float(previous_span["bbox"][2])
        current_x0 = float(span["bbox"][0])

        font_size = max(
            float(previous_span.get("size", 0.0)),
            float(span.get("size", 0.0)),
            1.0,
        )

        geometric_gap = current_x0 - previous_x1

        explicit_space = (
            previous_raw.endswith((" ", "\t"))
            or raw_text.startswith((" ", "\t"))
        )

        needs_space = explicit_space or geometric_gap > font_size * 0.08

        if (
            piece[0] in _CLOSING_PUNCTUATION
            or result.endswith(_NO_SPACE_AFTER)
        ):
            needs_space = False

        result += (" " if needs_space else "") + piece
        previous_span = span

    return normalize_whitespace(result)


def union_bbox(first: BoundingBox, second: BoundingBox) -> BoundingBox:
    """Return the bounding box containing both inputs."""
    return BoundingBox(
        x0=min(first.x0, second.x0),
        y0=min(first.y0, second.y0),
        x1=max(first.x1, second.x1),
        y1=max(first.y1, second.y1),
    )


def join_text(left: str, right: str) -> str:
    """Join adjacent extracted lines while repairing simple hyphenation."""
    left = left.rstrip()
    right = right.lstrip()

    if not left:
        return right

    if not right:
        return left

    if left.endswith("-") and right[0].islower():
        return left + right

    return f"{left} {right}"


def is_bold_span(span: dict) -> bool:
    """Infer bold text from the PDF font name."""
    font_name = str(span.get("font", "")).casefold()

    return "bold" in font_name or "black" in font_name


def extract_page_lines(
    page: pymupdf.Page,
    page_number: int,
) -> list[TextLine]:
    """Extract text lines in PyMuPDF's native document order."""
    page_dict = page.get_text("dict", sort=False)

    extracted_lines: list[TextLine] = []

    for block_index, block in enumerate(page_dict.get("blocks", [])):
        lines = block.get("lines")

        if not lines:
            continue

        for line_index, line in enumerate(lines):
            spans = [
                span
                for span in line.get("spans", [])
                if normalize_whitespace(str(span.get("text", "")))
            ]

            if not spans:
                continue

            text = join_spans(spans)

            if not text:
                continue

            x0, y0, x1, y1 = line["bbox"]

            font_size = max(
                float(span.get("size", 0.0))
                for span in spans
            )

            is_bold = any(
                is_bold_span(span)
                for span in spans
            )

            extracted_lines.append(
                TextLine(
                    text=text,
                    page_number=page_number,
                    block_index=block_index,
                    line_index=line_index,
                    bbox=BoundingBox(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                    ),
                    font_size=font_size,
                    is_bold=is_bold,
                )
            )

    return extracted_lines


def infer_body_font_size(lines: list[TextLine]) -> float:
    """Infer the document's dominant body font using character-weighted frequency."""
    weighted_sizes: Counter[float] = Counter()

    for line in lines:
        rounded_size = round(line.font_size, 1)
        weighted_sizes[rounded_size] += len(line.text)

    if not weighted_sizes:
        raise ValueError("Unable to infer body font size from an empty document.")

    return weighted_sizes.most_common(1)[0][0]


def find_repeated_boilerplate(
    lines: list[TextLine],
) -> set[str]:
    """Find repeated text occurring in header/footer regions."""
    pages_by_text: dict[str, set[int]] = {}

    for line in lines:
        in_header = line.bbox.y1 <= _HEADER_MAX_Y
        in_footer = line.bbox.y0 >= _FOOTER_MIN_Y

        if not (in_header or in_footer):
            continue

        normalized = normalize_boilerplate_text(line.text)

        if not normalized:
            continue

        pages_by_text.setdefault(normalized, set()).add(
            line.page_number
        )

    return {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= _MIN_REPEATED_BOILERPLATE_PAGES
    }


def is_boilerplate_line(
    line: TextLine,
    repeated_boilerplate: set[str],
) -> bool:
    """Return whether a line should be excluded from semantic document content."""
    text = normalize_whitespace(line.text)

    if _PAGE_NUMBER_PATTERN.fullmatch(text):
        return True

    normalized = normalize_boilerplate_text(text)

    in_header = line.bbox.y1 <= _HEADER_MAX_Y
    in_footer = line.bbox.y0 >= _FOOTER_MIN_Y

    return (
        normalized in repeated_boilerplate
        and (in_header or in_footer)
    )


def is_heading_line(
    line: TextLine,
    body_font_size: float,
) -> bool:
    """Classify likely headings using document-relative typography."""
    text = line.text.strip()

    if not text:
        return False

    if len(text) > 140:
        return False

    if text.startswith(_BULLET_PREFIXES):
        return False

    noticeably_larger = line.font_size >= body_font_size + 1.4

    moderately_larger_and_bold = (
        line.font_size >= body_font_size + 0.4
        and line.is_bold
    )

    return noticeably_larger or moderately_larger_and_bold


def is_list_item_line(line: TextLine) -> bool:
    """Return whether a line begins a visible list item."""
    return line.text.lstrip().startswith(_BULLET_PREFIXES)


def strip_bullet_prefix(text: str) -> str:
    """Remove one supported visual bullet prefix."""
    stripped = text.lstrip()

    for prefix in _BULLET_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()

    return stripped


def reconstruct_units(
    lines: list[TextLine],
    body_font_size: float,
    repeated_boilerplate: set[str],
) -> list[DocumentUnit]:
    """Reconstruct headings, paragraphs, and list items from extracted lines."""
    units: list[DocumentUnit] = []

    current_text = ""
    current_type: UnitType | None = None
    current_page: int | None = None
    current_bbox: BoundingBox | None = None
    current_font_size = body_font_size
    current_block: int | None = None

    source_order = 0

    def flush_current() -> None:
        nonlocal current_text
        nonlocal current_type
        nonlocal current_page
        nonlocal current_bbox
        nonlocal current_font_size
        nonlocal current_block
        nonlocal source_order

        if (
            not current_text
            or current_type is None
            or current_page is None
            or current_bbox is None
        ):
            return

        units.append(
            DocumentUnit(
                text=current_text.strip(),
                unit_type=current_type,
                page_number=current_page,
                bbox=current_bbox,
                source_order=source_order,
                font_size=current_font_size,
            )
        )

        source_order += 1

        current_text = ""
        current_type = None
        current_page = None
        current_bbox = None
        current_font_size = body_font_size
        current_block = None

    for line in lines:
        if is_boilerplate_line(
            line=line,
            repeated_boilerplate=repeated_boilerplate,
        ):
            continue

        if is_heading_line(
            line=line,
            body_font_size=body_font_size,
        ):
            flush_current()

            units.append(
                DocumentUnit(
                    text=line.text,
                    unit_type="heading",
                    page_number=line.page_number,
                    bbox=line.bbox,
                    source_order=source_order,
                    font_size=line.font_size,
                )
            )

            source_order += 1
            continue

        if is_list_item_line(line):
            flush_current()

            current_text = strip_bullet_prefix(line.text)
            current_type = "list_item"
            current_page = line.page_number
            current_bbox = line.bbox
            current_font_size = line.font_size
            current_block = line.block_index
            continue

        same_active_unit = (
            current_type in {"paragraph", "list_item"}
            and current_page == line.page_number
            and current_block == line.block_index
        )

        if same_active_unit:
            current_text = join_text(
                current_text,
                line.text,
            )

            if current_bbox is not None:
                current_bbox = union_bbox(
                    current_bbox,
                    line.bbox,
                )

            continue

        flush_current()

        current_text = line.text
        current_type = "paragraph"
        current_page = line.page_number
        current_bbox = line.bbox
        current_font_size = line.font_size
        current_block = line.block_index

    flush_current()

    return units


def parse_pdf(
    pdf_path: Path,
    document_id: str = "medicare",
) -> ParsedDocument:
    """Parse a PDF into page-aware semantic structural units."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    all_lines: list[TextLine] = []

    with pymupdf.open(pdf_path) as document:
        page_count = document.page_count

        for page_index, page in enumerate(
            document,
            start=1,
        ):
            all_lines.extend(
                extract_page_lines(
                    page=page,
                    page_number=page_index,
                )
            )

    if not all_lines:
        raise ValueError(
            "No extractable text was found in the supplied PDF."
        )

    body_font_size = infer_body_font_size(all_lines)

    repeated_boilerplate = find_repeated_boilerplate(
        all_lines
    )

    units = reconstruct_units(
        lines=all_lines,
        body_font_size=body_font_size,
        repeated_boilerplate=repeated_boilerplate,
    )

    return ParsedDocument(
        document_id=document_id,
        document_sha256=calculate_sha256(pdf_path),
        page_count=page_count,
        body_font_size=body_font_size,
        units=tuple(units),
    )