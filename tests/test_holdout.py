from pathlib import Path

from app.rag.evaluation import (
    gold_source_units,
    load_golden_queries,
)

GOLDEN_PATH = Path(
    "evaluation/golden_queries.json"
)

HOLDOUT_PATH = Path(
    "evaluation/holdout_queries.json"
)


def test_holdout_contains_five_queries() -> None:
    queries = load_golden_queries(
        HOLDOUT_PATH
    )

    assert len(queries) == 5


def test_holdout_ids_do_not_overlap_selection_queries() -> None:
    selection = load_golden_queries(
        GOLDEN_PATH
    )

    holdout = load_golden_queries(
        HOLDOUT_PATH
    )

    selection_ids = {
        query.query_id
        for query in selection
    }

    holdout_ids = {
        query.query_id
        for query in holdout
    }

    assert (
        selection_ids
        .isdisjoint(
            holdout_ids
        )
    )


def test_holdout_query_texts_do_not_overlap_selection_queries() -> None:
    selection = load_golden_queries(
        GOLDEN_PATH
    )

    holdout = load_golden_queries(
        HOLDOUT_PATH
    )

    selection_queries = {
        query.query.casefold()
        for query in selection
    }

    holdout_queries = {
        query.query.casefold()
        for query in holdout
    }

    assert (
        selection_queries
        .isdisjoint(
            holdout_queries
        )
    )


def test_holdout_labels_are_frozen() -> None:
    queries = load_golden_queries(
        HOLDOUT_PATH
    )

    expected_units = {
        "h01": frozenset(
            {
                908,
                909,
            }
        ),
        "h02": frozenset(
            {
                930,
            }
        ),
        "h03": frozenset(
            {
                1454,
                1455,
            }
        ),
        "h04": frozenset(
            {
                1246,
                1247,
            }
        ),
        "h05": frozenset(
            {
                944,
                945,
            }
        ),
    }

    actual_units = {
        query.query_id: (
            gold_source_units(
                query
            )
        )
        for query in queries
    }

    assert (
        actual_units
        == expected_units
    )


def test_holdout_gold_pages_are_frozen() -> None:
    queries = load_golden_queries(
        HOLDOUT_PATH
    )

    expected_pages = {
        "h01": (
            53,
        ),
        "h02": (
            54,
        ),
        "h03": (
            91,
        ),
        "h04": (
            75,
        ),
        "h05": (
            55,
        ),
    }

    actual_pages = {
        query.query_id: (
            query.gold_pages
        )
        for query in queries
    }

    assert (
        actual_pages
        == expected_pages
    )
