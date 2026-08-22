from pathlib import Path

from app.rag.evaluation import load_golden_queries
from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")
GOLD_PATH = Path("evaluation/golden_queries.json")


def main() -> None:
    """Verify every gold source-unit label exists in the document."""
    document = parse_pdf(PDF_PATH)
    queries = load_golden_queries(GOLD_PATH)

    units_by_order = {
        unit.source_order: unit
        for unit in document.units
    }

    for query in queries:
        for group in query.evidence_groups:
            for source_order in group:
                if source_order not in units_by_order:
                    raise RuntimeError(
                        f"{query.query_id}: unknown source "
                        f"order {source_order}"
                    )

        actual_pages = {
            units_by_order[source_order].page_number
            for group in query.evidence_groups
            for source_order in group
        }

        if not actual_pages.issubset(
            set(query.gold_pages)
        ):
            raise RuntimeError(
                f"{query.query_id}: evidence pages "
                f"{sorted(actual_pages)} do not match "
                f"gold pages {list(query.gold_pages)}"
            )

        print(
            f"{query.query_id}: PASS | "
            f"groups={len(query.evidence_groups)} | "
            f"pages={query.gold_pages}"
        )

    print()
    print(
        f"Validated {len(queries)} positive "
        "golden queries."
    )


if __name__ == "__main__":
    main()