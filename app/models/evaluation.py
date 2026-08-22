from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """A manually verified positive retrieval evaluation query."""

    query_id: str
    query: str
    category: str
    evidence_groups: tuple[tuple[int, ...], ...]
    gold_pages: tuple[int, ...]