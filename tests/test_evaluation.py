import json

import pytest

from app.models.document import DocumentChunk
from app.models.evaluation import GoldenQuery
from app.rag.evaluation import (
    group_recall_at_k,
    load_golden_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.rag.vector_store import SearchHit


def write_payload(
    tmp_path,
    payload: dict,
):
    path = tmp_path / "golden.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return path


def test_load_golden_queries(tmp_path) -> None:
    path = write_payload(
        tmp_path,
        {
            "queries": [
                {
                    "query_id": "q01",
                    "query": "What does Part B cover?",
                    "category": "coverage",
                    "evidence_groups": [
                        [10],
                        [11, 12],
                    ],
                    "gold_pages": [29],
                }
            ]
        },
    )

    queries = load_golden_queries(path)

    assert len(queries) == 1
    assert queries[0].query_id == "q01"
    assert queries[0].evidence_groups == (
        (10,),
        (11, 12),
    )


def test_duplicate_query_ids_are_rejected(
    tmp_path,
) -> None:
    item = {
        "query_id": "q01",
        "query": "Question",
        "category": "test",
        "evidence_groups": [[1]],
        "gold_pages": [1],
    }

    path = write_payload(
        tmp_path,
        {
            "queries": [
                item,
                item,
            ]
        },
    )

    with pytest.raises(ValueError):
        load_golden_queries(path)


def test_empty_evidence_group_is_rejected(
    tmp_path,
) -> None:
    path = write_payload(
        tmp_path,
        {
            "queries": [
                {
                    "query_id": "q01",
                    "query": "Question",
                    "category": "test",
                    "evidence_groups": [[]],
                    "gold_pages": [1],
                }
            ]
        },
    )

    with pytest.raises(ValueError):
        load_golden_queries(path)

def make_eval_chunk(
    chunk_id: str,
    source_orders: tuple[int, ...],
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        strategy_id="target_192",
        target_tokens=192,
        token_count=20,
        text=f"Text for {chunk_id}",
        heading_context=(),
        page_numbers=(1,),
        page_start=1,
        page_end=1,
        section_index=0,
        chunk_index=0,
        source_unit_orders=source_orders,
    )


def make_hit(
    rank: int,
    chunk: DocumentChunk,
) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0 / rank,
        chunk=chunk,
    )


def make_golden_query() -> GoldenQuery:
    return GoldenQuery(
        query_id="q01",
        query="Test query",
        category="test",
        evidence_groups=(
            (10,),
            (20, 21),
        ),
        gold_pages=(1,),
    )


def test_retrieval_metrics_measure_evidence_coverage() -> None:
    query = make_golden_query()

    irrelevant = make_eval_chunk(
        "irrelevant",
        (99,),
    )

    first_relevant = make_eval_chunk(
        "relevant-1",
        (10,),
    )

    second_relevant = make_eval_chunk(
        "relevant-2",
        (20,),
    )

    hits = (
        make_hit(1, irrelevant),
        make_hit(2, first_relevant),
        make_hit(3, second_relevant),
    )

    assert precision_at_k(
        hits,
        query,
        3,
    ) == pytest.approx(
        2 / 3
    )

    assert recall_at_k(
        hits,
        query,
        3,
    ) == pytest.approx(
        2 / 3
    )

    assert group_recall_at_k(
        hits,
        query,
        3,
    ) == pytest.approx(
        0.75
    )

    assert reciprocal_rank_at_k(
        hits,
        query,
        3,
    ) == pytest.approx(
        0.5
    )


def test_metrics_return_zero_when_no_evidence_is_retrieved() -> None:
    query = make_golden_query()

    chunk = make_eval_chunk(
        "irrelevant",
        (99,),
    )

    hits = (
        make_hit(1, chunk),
    )

    assert precision_at_k(
        hits,
        query,
        1,
    ) == 0.0

    assert recall_at_k(
        hits,
        query,
        1,
    ) == 0.0

    assert group_recall_at_k(
        hits,
        query,
        1,
    ) == 0.0

    assert reciprocal_rank_at_k(
        hits,
        query,
        1,
    ) == 0.0


def test_ndcg_rewards_better_ranking() -> None:
    query = make_golden_query()

    relevant_a = make_eval_chunk(
        "relevant-a",
        (10,),
    )

    relevant_b = make_eval_chunk(
        "relevant-b",
        (20,),
    )

    irrelevant = make_eval_chunk(
        "irrelevant",
        (99,),
    )

    corpus = (
        relevant_a,
        relevant_b,
        irrelevant,
    )

    better_hits = (
        make_hit(1, relevant_a),
        make_hit(2, relevant_b),
        make_hit(3, irrelevant),
    )

    worse_hits = (
        make_hit(1, irrelevant),
        make_hit(2, relevant_a),
        make_hit(3, relevant_b),
    )

    better_score = ndcg_at_k(
        better_hits,
        query,
        corpus,
        3,
    )

    worse_score = ndcg_at_k(
        worse_hits,
        query,
        corpus,
        3,
    )

    assert better_score > worse_score

    assert better_score == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
    ],
)
def test_metrics_reject_invalid_k(
    k: int,
) -> None:
    query = make_golden_query()

    with pytest.raises(ValueError):
        recall_at_k(
            (),
            query,
            k,
        )