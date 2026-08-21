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