import json
from math import inf

import pytest

from app.rag.evaluation import (
    evaluate_threshold,
    load_negative_queries,
    select_relevance_threshold,
)


def write_negative_payload(
    tmp_path,
    payload: dict,
):
    path = (
        tmp_path
        / "negative_queries.json"
    )

    path.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    return path


def test_load_negative_queries(
    tmp_path,
) -> None:
    path = write_negative_payload(
        tmp_path,
        {
            "queries": [
                {
                    "query_id": "n01",
                    "query": (
                        "What is the capital "
                        "of France?"
                    ),
                    "category": (
                        "unrelated"
                    ),
                }
            ]
        },
    )

    queries = (
        load_negative_queries(
            path
        )
    )

    assert len(
        queries
    ) == 1

    assert (
        queries[0].query_id
        == "n01"
    )

    assert (
        queries[0].category
        == "unrelated"
    )


def test_duplicate_negative_query_ids_are_rejected(
    tmp_path,
) -> None:
    item = {
        "query_id": "n01",
        "query": "Question",
        "category": "test",
    }

    path = write_negative_payload(
        tmp_path,
        {
            "queries": [
                item,
                item,
            ]
        },
    )

    with pytest.raises(
        ValueError
    ):
        load_negative_queries(
            path
        )


def test_empty_negative_query_is_rejected(
    tmp_path,
) -> None:
    path = write_negative_payload(
        tmp_path,
        {
            "queries": [
                {
                    "query_id": "n01",
                    "query": " ",
                    "category": "test",
                }
            ]
        },
    )

    with pytest.raises(
        ValueError
    ):
        load_negative_queries(
            path
        )


def test_evaluate_threshold_classifies_scores() -> None:
    result = evaluate_threshold(
        positive_scores=(
            0.80,
            0.90,
        ),
        negative_scores=(
            0.40,
            0.60,
        ),
        threshold=0.70,
    )

    assert (
        result.true_positive
        == 2
    )

    assert (
        result.false_negative
        == 0
    )

    assert (
        result.true_negative
        == 2
    )

    assert (
        result.false_positive
        == 0
    )

    assert (
        result.positive_recall
        == pytest.approx(
            1.0
        )
    )

    assert (
        result.negative_specificity
        == pytest.approx(
            1.0
        )
    )

    assert (
        result.balanced_accuracy
        == pytest.approx(
            1.0
        )
    )


def test_threshold_selection_finds_clean_separator() -> None:
    selection = (
        select_relevance_threshold(
            positive_scores=(
                0.80,
                0.90,
            ),
            negative_scores=(
                0.40,
                0.60,
            ),
        )
    )

    assert (
        selection.selected.threshold
        == pytest.approx(
            0.70
        )
    )

    assert (
        selection.selected.balanced_accuracy
        == pytest.approx(
            1.0
        )
    )


def test_threshold_selection_uses_deterministic_tie_break() -> None:
    selection = (
        select_relevance_threshold(
            positive_scores=(
                0.90,
                0.70,
            ),
            negative_scores=(
                0.80,
                0.40,
            ),
        )
    )

    assert (
        selection.selected.threshold
        == pytest.approx(
            0.55
        )
    )

    assert (
        selection.selected.positive_recall
        == pytest.approx(
            1.0
        )
    )

    assert (
        selection.selected.balanced_accuracy
        == pytest.approx(
            0.75
        )
    )


@pytest.mark.parametrize(
    (
        "positive_scores",
        "negative_scores",
    ),
    [
        (
            (),
            (0.5,),
        ),
        (
            (0.5,),
            (),
        ),
    ],
)
def test_threshold_selection_rejects_empty_classes(
    positive_scores,
    negative_scores,
) -> None:
    with pytest.raises(
        ValueError
    ):
        select_relevance_threshold(
            positive_scores,
            negative_scores,
        )


def test_threshold_selection_rejects_non_finite_scores() -> None:
    with pytest.raises(
        ValueError
    ):
        select_relevance_threshold(
            positive_scores=(
                0.8,
                inf,
            ),
            negative_scores=(
                0.4,
            ),
        )