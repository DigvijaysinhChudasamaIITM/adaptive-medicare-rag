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


@dataclass(frozen=True, slots=True)
class NegativeQuery:
    """A deliberately unsupported query used for threshold calibration."""

    query_id: str
    query: str
    category: str


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Binary classification metrics for one relevance threshold."""

    threshold: float
    true_positive: int
    false_negative: int
    true_negative: int
    false_positive: int
    positive_recall: float
    negative_specificity: float
    balanced_accuracy: float


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Result of deterministic threshold selection."""

    selected: ThresholdMetrics
    candidates_evaluated: int