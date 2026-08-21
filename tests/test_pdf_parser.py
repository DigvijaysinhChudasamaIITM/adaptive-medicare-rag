from pathlib import Path

import pytest

from app.rag.pdf_parser import parse_pdf

PDF_PATH = Path("data/medicare.pdf")


@pytest.fixture(scope="module")
def parsed_document():
    return parse_pdf(PDF_PATH)


def test_parser_preserves_physical_page_count(
    parsed_document,
) -> None:
    assert parsed_document.page_count == 128


def test_parser_infers_expected_body_font(
    parsed_document,
) -> None:
    assert parsed_document.body_font_size == pytest.approx(
        11.0,
        abs=0.1,
    )


def test_parser_preserves_page_metadata(
    parsed_document,
) -> None:
    page_17_units = [
        unit
        for unit in parsed_document.units
        if unit.page_number == 17
    ]

    assert page_17_units

    assert all(
        unit.page_number == 17
        for unit in page_17_units
    )


def test_page_numbers_are_removed_from_content(
    parsed_document,
) -> None:
    standalone_page_numbers = {
        unit.text
        for unit in parsed_document.units
        if unit.text.isdigit()
    }

    assert standalone_page_numbers == set()


def test_initial_enrollment_heading_is_detected(
    parsed_document,
) -> None:
    matching_units = [
        unit
        for unit in parsed_document.units
        if unit.text == "Initial Enrollment Period"
    ]

    assert matching_units

    assert matching_units[0].unit_type == "heading"
    assert matching_units[0].page_number == 17


def test_initial_enrollment_content_is_preserved(
    parsed_document,
) -> None:
    page_17_text = " ".join(
        unit.text
        for unit in parsed_document.units
        if unit.page_number == 17
    )

    assert "7-month period" in page_17_text
    assert "3 months before" in page_17_text
    assert "3 months after" in page_17_text


def test_open_enrollment_list_item_is_preserved(
    parsed_document,
) -> None:
    page_80_units = [
        unit
        for unit in parsed_document.units
        if unit.page_number == 80
    ]

    matching_units = [
        unit
        for unit in page_80_units
        if "Open Enrollment Period" in unit.text
        and "October 15" in unit.text
        and "December 7" in unit.text
    ]

    assert matching_units

    assert matching_units[0].unit_type == "list_item"


def test_blank_page_has_no_semantic_units(
    parsed_document,
) -> None:
    page_127_units = [
        unit
        for unit in parsed_document.units
        if unit.page_number == 127
    ]

    assert page_127_units == []


def test_notes_page_does_not_create_page_number_unit(
    parsed_document,
) -> None:
    page_118_texts = {
        unit.text
        for unit in parsed_document.units
        if unit.page_number == 118
    }

    assert "118" not in page_118_texts

def test_span_boundaries_preserve_word_spacing(
    parsed_document,
) -> None:
    page_11_text = " ".join(
        unit.text
        for unit in parsed_document.units
        if unit.page_number == 11
    )

    assert "amount after" in page_11_text
    assert "a yearly limit" in page_11_text
    assert "You can choose" in page_11_text


def test_callout_span_boundaries_preserve_spaces(
    parsed_document,
) -> None:
    page_17_text = " ".join(
        unit.text
        for unit in parsed_document.units
        if unit.page_number == 17
    )

    page_80_text = " ".join(
        unit.text
        for unit in parsed_document.units
        if unit.page_number == 80
    )

    assert "Important! Remember" in page_17_text
    assert "Important! If" in page_80_text