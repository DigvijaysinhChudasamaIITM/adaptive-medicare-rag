from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")


@dataclass(frozen=True, slots=True)
class EvidenceProbe:
    """Known document phrase used to locate gold evidence."""

    query_id: str
    query: str
    phrase: str


PROBES = (
    EvidenceProbe(
        query_id="q01",
        query=(
            "What are the requirements to join "
            "a Medicare Advantage Plan?"
        ),
        phrase="To join a Medicare Advantage Plan, you must",
    ),
    EvidenceProbe(
        query_id="q02",
        query=(
            "How is the Medicare Part D late "
            "enrollment penalty calculated?"
        ),
        phrase=(
            "How much more will I pay for a late "
            "enrollment penalty?"
        ),
    ),
    EvidenceProbe(
        query_id="q03",
        query="How do I file a Medicare appeal?",
        phrase=(
            "How you file an appeal depends on the "
            "type of Medicare coverage you have"
        ),
    ),
    EvidenceProbe(
        query_id="q04",
        query=(
            "When can I first sign up for Medicare?"
        ),
        phrase="Initial Enrollment Period",
    ),
    EvidenceProbe(
        query_id="q05",
        query=(
            "What happens if I miss my Medicare "
            "Initial Enrollment Period?"
        ),
        phrase="General Enrollment Period",
    ),
    EvidenceProbe(
        query_id="q06",
        query=(
            "What does Medicare Part B generally cover?"
        ),
        phrase="What does Part B cover?",
    ),
    EvidenceProbe(
        query_id="q07",
        query=(
            "What is Medicare drug coverage Part D?"
        ),
        phrase="Medicare drug coverage (Part D)",
    ),
    EvidenceProbe(
        query_id="q08",
        query=(
            "When can I join or switch a Medicare "
            "Advantage Plan?"
        ),
        phrase=(
            "You can join, switch, drop, or make changes"
        ),
    ),
    EvidenceProbe(
        query_id="q09",
        query=(
            "Can I have Medigap while I am in a "
            "Medicare Advantage Plan?"
        ),
        phrase=(
            "You can’t buy Medigap while you’re in "
            "a Medicare Advantage Plan"
        ),
    ),
    EvidenceProbe(
        query_id="q10",
        query=(
            "How can I avoid the Medicare Part D "
            "late enrollment penalty?"
        ),
        phrase="There are 3 ways to avoid",
    ),
    EvidenceProbe(
        query_id="q11",
        query=(
            "Can someone represent me in a "
            "Medicare appeal?"
        ),
        phrase="You can appoint a representative",
    ),
    EvidenceProbe(
        query_id="q12",
        query=(
            "What happens to Medicare Advantage "
            "coverage if a plan leaves Medicare?"
        ),
        phrase=(
            "If the plan decides to stop participating "
            "in Medicare"
        ),
    ),
)


def main() -> None:
    """Locate independent gold-evidence candidates."""
    document = parse_pdf(PDF_PATH)

    for probe in PROBES:
        print()
        print("=" * 100)
        print(f"{probe.query_id}: {probe.query}")
        print(f"ANCHOR: {probe.phrase}")
        print("=" * 100)

        phrase = probe.phrase.casefold()

        matches = [
            unit
            for unit in document.units
            if phrase in unit.text.casefold()
        ]

        if not matches:
            print("NO EXACT MATCH")
            continue

        for unit in matches:
            print(
                f"order={unit.source_order:<4} "
                f"page={unit.page_number:<3} "
                f"type={unit.unit_type:<10}"
            )
            print(unit.text)
            print()


if __name__ == "__main__":
    main()