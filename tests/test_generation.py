import pytest
from pydantic import ValidationError

from app.models.generation import GeneratedAnswer, GroundingEvidence


def test_grounding_evidence_strips_values() -> None:
    evidence = GroundingEvidence(
        chunk_id="  chunk-1  ",
        text="  Medicare evidence.  ",
    )

    assert evidence.chunk_id == "chunk-1"
    assert evidence.text == "Medicare evidence."


def test_grounding_evidence_rejects_whitespace_only_text() -> None:
    with pytest.raises(ValidationError):
        GroundingEvidence(
            chunk_id="chunk-1",
            text="   ",
        )


def test_generated_answer_accepts_supported_answer() -> None:
    answer = GeneratedAnswer(
        answer="  Medicare covers this service.  ",
        citations=["  chunk-1  "],
    )

    assert answer.answer == "Medicare covers this service."
    assert answer.citations == ["chunk-1"]


def test_generated_answer_allows_empty_citations_for_abstention() -> None:
    answer = GeneratedAnswer(
        answer="Insufficient evidence.",
        citations=[],
    )

    assert answer.citations == []


def test_generated_answer_rejects_blank_citation_id() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer(
            answer="Supported answer.",
            citations=["   "],
        )


def test_generated_answer_forbids_extra_llm_authored_metadata() -> None:
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(
            {
                "answer": "Supported answer.",
                "citations": ["chunk-1"],
                "page_number": 42,
            }
        )