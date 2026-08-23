from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path(
    "data/medicare.pdf"
)

CONTEXT_RADIUS = 2


@dataclass(
    frozen=True,
    slots=True,
)
class HoldoutCandidate:
    query_id: str
    query: str
    category: str
    anchors: tuple[str, ...]


CANDIDATES = (
    HoldoutCandidate(
        query_id="h01",
        query=(
            "Does Medicare cover health care "
            "while I am traveling outside the U.S.?"
        ),
        category="travel_coverage",
        anchors=(
            "Medicare generally doesn’t cover health care "
            "while you’re traveling outside the U.S.",
            "Travel",
        ),
    ),
    HoldoutCandidate(
        query_id="h02",
        query=(
            "Who can get a yearly Medicare Wellness visit, "
            "and how often is it covered?"
        ),
        category="preventive_services",
        anchors=(
            "Yearly “Wellness” visit",
            "If you’ve had Part B for longer than 12 months",
        ),
    ),
    HoldoutCandidate(
        query_id="h03",
        query=(
            "What are the Medicare Savings Programs "
            "and what costs can they help pay?"
        ),
        category="financial_assistance",
        anchors=(
            "Medicare Savings Programs",
            "There are 4 kinds of Medicare Savings Programs",
        ),
    ),
    HoldoutCandidate(
        query_id="h04",
        query=(
            "What does Medigap help pay for, "
            "and what does it generally not cover?"
        ),
        category="medigap",
        anchors=(
            "How does Medigap work?",
            "Generally, Medigap",
        ),
    ),
    HoldoutCandidate(
        query_id="h05",
        query=(
            "What are some services that Original Medicare "
            "does not cover?"
        ),
        category="coverage_exclusions",
        anchors=(
            "What ISN'T covered by Part A and Part B?",
            "Some of the items and services that "
            "Original Medicare doesn’t cover",
        ),
    ),
)


def normalized(
    value: str,
) -> str:
    """Normalize text for case-insensitive literal inspection."""
    return " ".join(
        value.casefold().split()
    )


def main() -> None:
    """Print direct PDF-parser evidence windows for holdout labeling."""
    document = parse_pdf(
        PDF_PATH
    )

    units = document.units

    print(
        "document_sha256:",
        document.document_sha256,
    )
    print(
        "page_count:",
        document.page_count,
    )
    print(
        "unit_count:",
        len(units),
    )

    for candidate in CANDIDATES:
        print()
        print(
            "=" * 100
        )
        print(
            f"{candidate.query_id}: "
            f"{candidate.query}"
        )
        print(
            f"CATEGORY: {candidate.category}"
        )
        print(
            "=" * 100
        )

        matching_indices: list[int] = []

        for index, unit in enumerate(
            units
        ):
            unit_text = normalized(
                unit.text
            )

            if any(
                normalized(anchor)
                in unit_text
                for anchor
                in candidate.anchors
            ):
                matching_indices.append(
                    index
                )

        if not matching_indices:
            print(
                "NO DIRECT ANCHOR MATCH"
            )
            continue

        printed_orders: set[int] = set()

        for match_index in matching_indices:
            start = max(
                0,
                match_index
                - CONTEXT_RADIUS,
            )

            end = min(
                len(units),
                match_index
                + CONTEXT_RADIUS
                + 1,
            )

            print()
            print(
                f"--- match around "
                f"source order "
                f"{units[match_index].source_order} ---"
            )

            for unit in units[
                start:end
            ]:
                if (
                    unit.source_order
                    in printed_orders
                ):
                    continue

                printed_orders.add(
                    unit.source_order
                )

                print(
                    f"order={unit.source_order:<4} "
                    f"page={unit.page_number:<3} "
                    f"type={unit.unit_type:<10}"
                )

                print(
                    unit.text
                )
                print()


if __name__ == "__main__":
    main()