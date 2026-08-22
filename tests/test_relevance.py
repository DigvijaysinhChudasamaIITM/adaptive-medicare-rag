import json

import pytest

from app.models.document import DocumentChunk
from app.rag.relevance import RelevanceGate
from app.rag.vector_store import SearchHit


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        strategy_id="target_416",
        target_tokens=416,
        token_count=100,
        text="Example Medicare evidence.",
        heading_context=(
            "Medicare",
        ),
        page_numbers=(10,),
        page_start=10,
        page_end=10,
        section_index=1,
        chunk_index=0,
        source_unit_orders=(1,),
    )


def make_hit(
    score: float,
) -> SearchHit:
    return SearchHit(
        rank=1,
        score=score,
        chunk=make_chunk(),
    )


def test_gate_rejects_no_hits() -> None:
    gate = RelevanceGate(
        threshold=0.70
    )

    decision = gate.assess(
        ()
    )

    assert (
        decision.is_relevant
        is False
    )

    assert (
        decision.top_score
        is None
    )

    assert (
        decision.reason
        == "no_hits"
    )


def test_gate_rejects_below_threshold() -> None:
    gate = RelevanceGate(
        threshold=0.70
    )

    decision = gate.assess(
        (
            make_hit(
                0.69
            ),
        )
    )

    assert (
        decision.is_relevant
        is False
    )

    assert (
        decision.reason
        == "below_threshold"
    )


def test_gate_accepts_score_equal_to_threshold() -> None:
    gate = RelevanceGate(
        threshold=0.70
    )

    decision = gate.assess(
        (
            make_hit(
                0.70
            ),
        )
    )

    assert (
        decision.is_relevant
        is True
    )

    assert (
        decision.reason
        == "relevant"
    )


def test_gate_accepts_score_above_threshold() -> None:
    gate = RelevanceGate(
        threshold=0.70
    )

    decision = gate.assess(
        (
            make_hit(
                0.85
            ),
        )
    )

    assert (
        decision.is_relevant
        is True
    )

    assert (
        decision.top_score
        == pytest.approx(
            0.85
        )
    )


def test_gate_rejects_non_finite_threshold() -> None:
    with pytest.raises(
        ValueError
    ):
        RelevanceGate(
            threshold=float(
                "nan"
            )
        )


def test_gate_rejects_non_finite_top_score() -> None:
    gate = RelevanceGate(
        threshold=0.70
    )

    with pytest.raises(
        ValueError
    ):
        gate.assess(
            (
                make_hit(
                    float(
                        "nan"
                    )
                ),
            )
        )


def test_gate_loads_threshold_from_artifact(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "relevance_calibration.json"
    )

    path.write_text(
        json.dumps(
            {
                "selected_threshold": (
                    0.73
                )
            }
        ),
        encoding="utf-8",
    )

    gate = RelevanceGate.load(
        path
    )

    assert (
        gate.threshold
        == pytest.approx(
            0.73
        )
    )

@pytest.mark.parametrize(
    "threshold",
    [
        -1.01,
        1.01,
    ],
)
def test_gate_rejects_out_of_range_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError
    ):
        RelevanceGate(
            threshold=threshold
        )