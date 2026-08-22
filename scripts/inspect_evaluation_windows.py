from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """A source-unit location requiring contextual inspection."""

    query_id: str
    query: str
    center_order: int
    radius: int = 4


WINDOWS = (
    EvidenceWindow(
        query_id="q01",
        query=(
            "What are the requirements to join "
            "a Medicare Advantage Plan?"
        ),
        center_order=1075,
        radius=5,
    ),
    EvidenceWindow(
        query_id="q02",
        query=(
            "How is the Medicare Part D late "
            "enrollment penalty calculated?"
        ),
        center_order=1373,
        radius=2,
    ),
    EvidenceWindow(
        query_id="q03",
        query="How do I file a Medicare appeal?",
        center_order=1577,
        radius=6,
    ),
    EvidenceWindow(
        query_id="q04",
        query="When can I first sign up for Medicare?",
        center_order=341,
        radius=6,
    ),
    EvidenceWindow(
        query_id="q05",
        query=(
            "What happens if I miss my Medicare "
            "Initial Enrollment Period?"
        ),
        center_order=357,
        radius=4,
    ),
    EvidenceWindow(
        query_id="q06",
        query="What does Medicare Part B generally cover?",
        center_order=531,
        radius=6,
    ),
    EvidenceWindow(
        query_id="q07",
        query="What is Medicare drug coverage Part D?",
        center_order=1311,
        radius=3,
    ),
    EvidenceWindow(
        query_id="q08",
        query=(
            "When can I join or switch a Medicare "
            "Advantage Plan?"
        ),
        center_order=1161,
        radius=7,
    ),
    EvidenceWindow(
        query_id="q09",
        query=(
            "Can I have Medigap while I am in a "
            "Medicare Advantage Plan?"
        ),
        center_order=1091,
        radius=2,
    ),
    EvidenceWindow(
        query_id="q10",
        query=(
            "How can I avoid the Medicare Part D "
            "late enrollment penalty?"
        ),
        center_order=1371,
        radius=3,
    ),
    EvidenceWindow(
        query_id="q11",
        query=(
            "Can someone represent me in a "
            "Medicare appeal?"
        ),
        center_order=1594,
        radius=2,
    ),
    EvidenceWindow(
        query_id="q12",
        query=(
            "What happens to Medicare Advantage "
            "coverage if a plan leaves Medicare?"
        ),
        center_order=1082,
        radius=2,
    ),
)


def main() -> None:
    """Print semantic-unit windows around proposed gold evidence."""
    document = parse_pdf(PDF_PATH)

    units_by_order = {
        unit.source_order: unit
        for unit in document.units
    }

    maximum_order = max(units_by_order)

    for window in WINDOWS:
        print()
        print("=" * 110)
        print(
            f"{window.query_id}: "
            f"{window.query}"
        )
        print(
            f"CENTER ORDER: "
            f"{window.center_order}"
        )
        print("=" * 110)

        start = max(
            0,
            window.center_order - window.radius,
        )

        end = min(
            maximum_order,
            window.center_order + window.radius,
        )

        for source_order in range(
            start,
            end + 1,
        ):
            unit = units_by_order.get(
                source_order
            )

            if unit is None:
                continue

            marker = (
                ">>>"
                if source_order
                == window.center_order
                else "   "
            )

            print(
                f"{marker} "
                f"order={unit.source_order:<4} "
                f"page={unit.page_number:<3} "
                f"type={unit.unit_type:<10}"
            )

            print(unit.text)
            print()


if __name__ == "__main__":
    main()