from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

UnitType = Literal["heading", "paragraph", "list_item"]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Coordinates of extracted content in PDF page space."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class TextLine:
    """A line extracted from the PDF with layout metadata."""

    text: str
    page_number: int
    block_index: int
    line_index: int
    bbox: BoundingBox
    font_size: float
    is_bold: bool


@dataclass(frozen=True, slots=True)
class DocumentUnit:
    """A semantic unit produced by PDF structural reconstruction."""

    text: str
    unit_type: UnitType
    page_number: int
    bbox: BoundingBox
    source_order: int
    font_size: float


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Structured representation of one parsed PDF."""

    document_id: str
    document_sha256: str
    page_count: int
    body_font_size: float
    units: tuple[DocumentUnit, ...]

@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrieval chunk constructed from semantic document units."""

    chunk_id: str
    strategy_id: str
    target_tokens: int
    token_count: int
    text: str
    heading_context: tuple[str, ...]
    page_numbers: tuple[int, ...]
    page_start: int
    page_end: int
    section_index: int
    chunk_index: int
    source_unit_orders: tuple[int, ...]