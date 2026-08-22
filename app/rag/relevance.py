from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from app.rag.vector_store import SearchHit

RelevanceReason = Literal[
    "relevant",
    "below_threshold",
    "no_hits",
]


@dataclass(
    frozen=True,
    slots=True,
)
class RelevanceDecision:
    """Deterministic relevance decision for retrieved evidence."""

    is_relevant: bool
    top_score: float | None
    threshold: float
    reason: RelevanceReason


@dataclass(
    frozen=True,
    slots=True,
)
class RelevanceGate:
    """Apply a calibrated relevance threshold to retrieval results."""

    threshold: float

    def __post_init__(
        self,
    ) -> None:
        if not isfinite(
            self.threshold
        ):
            raise ValueError(
                "Relevance threshold must be finite."
            )

        if not (
            -1.0
            <= self.threshold
            <= 1.0
        ):
            raise ValueError(
                "Relevance threshold must be "
                "between -1.0 and 1.0."
            )

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> RelevanceGate:
        """Load the selected threshold from a calibration artifact."""
        if not path.exists():
            raise FileNotFoundError(
                f"Relevance calibration artifact not found: {path}"
            )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        threshold = float(
            payload[
                "selected_threshold"
            ]
        )

        return cls(
            threshold=threshold
        )

    def assess(
        self,
        hits: Sequence[
            SearchHit
        ],
    ) -> RelevanceDecision:
        """Decide whether retrieval evidence is strong enough to use."""
        if not hits:
            return RelevanceDecision(
                is_relevant=False,
                top_score=None,
                threshold=(
                    self.threshold
                ),
                reason="no_hits",
            )

        top_score = float(
            hits[0].score
        )

        if not isfinite(
            top_score
        ):
            raise ValueError(
                "Top retrieval score must be finite."
            )

        if (
            top_score
            < self.threshold
        ):
            return RelevanceDecision(
                is_relevant=False,
                top_score=top_score,
                threshold=(
                    self.threshold
                ),
                reason=(
                    "below_threshold"
                ),
            )

        return RelevanceDecision(
            is_relevant=True,
            top_score=top_score,
            threshold=(
                self.threshold
            ),
            reason="relevant",
        )