import pytest

from app.models.document import (
    BoundingBox,
    DocumentUnit,
    UnitType,
)
from app.rag.chunking import (
    ChunkTargetStatistics,
    build_chunks,
    build_structural_sections,
    compose_chunk_text,
    derive_candidate_targets,
    snap_to_multiple,
)


def make_unit(
    text: str,
    unit_type: UnitType,
    source_order: int,
    page_number: int = 1,
) -> DocumentUnit:
    return DocumentUnit(
        text=text,
        unit_type=unit_type,
        page_number=page_number,
        bbox=BoundingBox(
            x0=0.0,
            y0=0.0,
            x1=100.0,
            y1=20.0,
        ),
        source_order=source_order,
        font_size=11.0,
    )

def test_consecutive_headings_across_pages_start_new_section() -> None:
    units = (
        make_unit(
            "What’s new &",
            "heading",
            0,
            page_number=2,
        ),
        make_unit(
            "important?",
            "heading",
            1,
            page_number=2,
        ),
        make_unit(
            "Contents",
            "heading",
            2,
            page_number=3,
        ),
        make_unit(
            "Index of topics",
            "paragraph",
            3,
            page_number=3,
        ),
    )

    sections = build_structural_sections(units)

    assert len(sections) == 2

    assert [
        unit.text
        for unit in sections[0]
    ] == [
        "What’s new &",
        "important?",
    ]

    assert [
        unit.text
        for unit in sections[1]
    ] == [
        "Contents",
        "Index of topics",
    ]

def test_consecutive_headings_remain_in_same_section() -> None:
    units = (
        make_unit(
            "Supplemental",
            "heading",
            0,
        ),
        make_unit(
            "coverage",
            "heading",
            1,
        ),
        make_unit(
            "Coverage explanation.",
            "paragraph",
            2,
        ),
        make_unit(
            "Next topic",
            "heading",
            3,
        ),
        make_unit(
            "Next explanation.",
            "paragraph",
            4,
        ),
    )

    sections = build_structural_sections(units)

    assert len(sections) == 2

    assert [
        unit.text
        for unit in sections[0]
    ] == [
        "Supplemental",
        "coverage",
        "Coverage explanation.",
    ]

    assert [
        unit.text
        for unit in sections[1]
    ] == [
        "Next topic",
        "Next explanation.",
    ]


def test_snap_to_multiple_uses_nearest_32_tokens() -> None:
    assert snap_to_multiple(121) == 128
    assert snap_to_multiple(188) == 192
    assert snap_to_multiple(332.2) == 320
    assert snap_to_multiple(414.8) == 416


def test_candidate_targets_are_document_derived() -> None:
    statistics = ChunkTargetStatistics(
        section_median=101,
        section_p75=188,
        section_p90=332.2,
        section_p95=414.8,
        paragraph_p95=121,
        model_max_length=512,
    )

    assert derive_candidate_targets(
        statistics
    ) == (
        128,
        192,
        320,
        416,
    )


def test_candidate_targets_are_deduplicated() -> None:
    statistics = ChunkTargetStatistics(
        section_median=110,
        section_p75=120,
        section_p90=125,
        section_p95=128,
        paragraph_p95=115,
        model_max_length=512,
    )

    assert derive_candidate_targets(
        statistics
    ) == (128,)


def test_candidate_targets_respect_model_limit() -> None:
    statistics = ChunkTargetStatistics(
        section_median=400,
        section_p75=500,
        section_p90=620,
        section_p95=900,
        paragraph_p95=450,
        model_max_length=512,
    )

    candidates = derive_candidate_targets(
        statistics
    )

    assert max(candidates) <= 512


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_snap_to_multiple_rejects_non_positive_values(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        snap_to_multiple(value)

class FakeTokenizer:
    model_max_length = 50

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        tokens = text.split()
        token_ids = list(range(len(tokens)))

        if add_special_tokens:
            return [-1, *token_ids, -2]

        return token_ids

def test_chunks_repeat_heading_context_for_continuations() -> None:
    tokenizer = FakeTokenizer()

    units = (
        make_unit(
            "Enrollment",
            "heading",
            0,
        ),
        make_unit(
            "one two three four five",
            "paragraph",
            1,
        ),
        make_unit(
            "six seven eight nine ten",
            "paragraph",
            2,
        ),
    )

    sections = build_structural_sections(units)

    chunks = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=9,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    assert len(chunks) == 2

    assert chunks[0].heading_context == (
        "Enrollment",
    )

    assert chunks[1].heading_context == (
        "Enrollment",
    )


def test_soft_target_does_not_split_semantic_unit() -> None:
    tokenizer = FakeTokenizer()

    long_paragraph = " ".join(
        f"word{i}"
        for i in range(20)
    )

    units = (
        make_unit(
            "Coverage",
            "heading",
            0,
        ),
        make_unit(
            long_paragraph,
            "paragraph",
            1,
        ),
    )

    sections = build_structural_sections(units)

    chunks = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=10,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    assert len(chunks) == 1
    assert chunks[0].token_count > 10
    assert chunks[0].token_count <= 50
    assert chunks[0].text == long_paragraph


def test_chunk_ids_are_deterministic_and_unique() -> None:
    tokenizer = FakeTokenizer()

    units = (
        make_unit(
            "Enrollment",
            "heading",
            0,
        ),
        make_unit(
            "one two three four five",
            "paragraph",
            1,
        ),
        make_unit(
            "six seven eight nine ten",
            "paragraph",
            2,
        ),
    )

    sections = build_structural_sections(units)

    first_run = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=9,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    second_run = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=9,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    first_ids = [
        chunk.chunk_id
        for chunk in first_run
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_run
    ]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_all_source_units_are_represented() -> None:
    tokenizer = FakeTokenizer()

    units = (
        make_unit(
            "Enrollment",
            "heading",
            0,
        ),
        make_unit(
            "first paragraph content",
            "paragraph",
            1,
        ),
        make_unit(
            "second paragraph content",
            "paragraph",
            2,
        ),
        make_unit(
            "Next topic",
            "heading",
            3,
        ),
        make_unit(
            "third paragraph content",
            "paragraph",
            4,
        ),
    )

    sections = build_structural_sections(units)

    chunks = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=10,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    represented_orders = {
        source_order
        for chunk in chunks
        for source_order in chunk.source_unit_orders
    }

    expected_orders = {
        unit.source_order
        for unit in units
    }

    assert represented_orders == expected_orders


def test_chunk_embedding_text_respects_hard_limit() -> None:
    tokenizer = FakeTokenizer()

    units = (
        make_unit(
            "Enrollment",
            "heading",
            0,
        ),
        make_unit(
            "one two three four",
            "paragraph",
            1,
        ),
        make_unit(
            "five six seven eight",
            "paragraph",
            2,
        ),
        make_unit(
            "nine ten eleven twelve",
            "paragraph",
            3,
        ),
    )

    sections = build_structural_sections(units)

    chunks = build_chunks(
        document_id="test",
        sections=sections,
        target_tokens=9,
        tokenizer=tokenizer,
        hard_limit=50,
    )

    for chunk in chunks:
        embedding_text = compose_chunk_text(
            chunk.heading_context,
            chunk.text,
        )

        assert len(
            tokenizer.encode(
                embedding_text,
                add_special_tokens=True,
                truncation=False,
            )
        ) <= 50
