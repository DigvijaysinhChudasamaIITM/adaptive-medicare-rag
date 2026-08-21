from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.models.document import (
    DocumentChunk,
    DocumentUnit,
)
from app.rag.tokenization import (
    TokenizerLike,
    count_tokens,
)


@dataclass(frozen=True, slots=True)
class ChunkTargetStatistics:
    """Document statistics used to derive candidate chunk targets."""

    section_median: float
    section_p75: float
    section_p90: float
    section_p95: float
    paragraph_p95: float
    model_max_length: int


def build_structural_sections(
    units: Sequence[DocumentUnit],
) -> list[tuple[DocumentUnit, ...]]:
    """Group semantic units into heading-led structural sections."""
    sections: list[tuple[DocumentUnit, ...]] = []

    current_section: list[DocumentUnit] = []
    current_has_content = False

    for unit in units:
        if unit.unit_type == "heading":
            page_changed = (
                bool(current_section)
                and unit.page_number
                != current_section[-1].page_number
            )

            if current_section and (
                current_has_content
                or page_changed
            ):
                sections.append(tuple(current_section))
                current_section = []
                current_has_content = False

            current_section.append(unit)
            continue

        current_section.append(unit)
        current_has_content = True

    if current_section:
        sections.append(tuple(current_section))

    return sections

def section_text(
    units: Sequence[DocumentUnit],
) -> str:
    """Create text for a reconstructed structural section."""
    return "\n".join(
        unit.text
        for unit in units
        if unit.text.strip()
    )


def snap_to_multiple(
    value: float,
    multiple: int = 32,
) -> int:
    """Round a positive token target to the nearest practical multiple."""
    if value <= 0:
        raise ValueError(
            "Chunk target values must be positive."
        )

    if multiple <= 0:
        raise ValueError(
            "Token multiple must be positive."
        )

    return max(
        multiple,
        int(
            math.floor(
                (value / multiple) + 0.5
            )
            * multiple
        ),
    )


def derive_candidate_targets(
    statistics: ChunkTargetStatistics,
) -> tuple[int, ...]:
    """Derive candidate chunk targets from document token statistics."""
    if statistics.model_max_length <= 0:
        raise ValueError(
            "Model maximum length must be positive."
        )

    raw_candidates = (
        max(
            statistics.section_median,
            statistics.paragraph_p95,
        ),
        statistics.section_p75,
        statistics.section_p90,
        statistics.section_p95,
    )

    snapped_candidates = {
        min(
            snap_to_multiple(value),
            statistics.model_max_length,
        )
        for value in raw_candidates
    }

    candidates = tuple(sorted(snapped_candidates))

    if not candidates:
        raise ValueError(
            "No candidate chunk targets could be derived."
        )

    return candidates


def compose_chunk_text(
    heading_context: Sequence[str],
    text: str,
) -> str:
    """Create the text that will later be embedded for a chunk."""
    parts = [
        heading.strip()
        for heading in heading_context
        if heading.strip()
    ]

    if text.strip():
        parts.append(text.strip())

    return "\n".join(parts)


def leading_heading_units(
    section: Sequence[DocumentUnit],
) -> tuple[DocumentUnit, ...]:
    """Return consecutive heading units at the start of a section."""
    headings: list[DocumentUnit] = []

    for unit in section:
        if unit.unit_type != "heading":
            break

        headings.append(unit)

    return tuple(headings)


def make_chunk(
    *,
    document_id: str,
    section_index: int,
    chunk_index: int,
    target_tokens: int,
    heading_units: Sequence[DocumentUnit],
    content_units: Sequence[DocumentUnit],
    tokenizer: TokenizerLike,
) -> DocumentChunk:
    """Create one deterministic document chunk."""
    if not content_units:
        raise ValueError(
            "A chunk must contain at least one content unit."
        )

    heading_context = tuple(
        unit.text
        for unit in heading_units
    )

    content_text = "\n".join(
        unit.text
        for unit in content_units
    )

    embedding_text = compose_chunk_text(
        heading_context,
        content_text,
    )

    token_count = count_tokens(
        embedding_text,
        tokenizer,
    )

    source_units = [
        *heading_units,
        *content_units,
    ]

    source_unit_orders = tuple(
        dict.fromkeys(
            unit.source_order
            for unit in source_units
        )
    )

    page_numbers = tuple(
        sorted(
            {
                unit.page_number
                for unit in source_units
            }
        )
    )

    strategy_id = f"target_{target_tokens}"

    chunk_id = (
        f"{document_id}"
        f"-t{target_tokens}"
        f"-s{section_index:04d}"
        f"-c{chunk_index:02d}"
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        strategy_id=strategy_id,
        target_tokens=target_tokens,
        token_count=token_count,
        text=content_text,
        heading_context=heading_context,
        page_numbers=page_numbers,
        page_start=min(page_numbers),
        page_end=max(page_numbers),
        section_index=section_index,
        chunk_index=chunk_index,
        source_unit_orders=source_unit_orders,
    )


def build_chunks(
    *,
    document_id: str,
    sections: Sequence[Sequence[DocumentUnit]],
    target_tokens: int,
    tokenizer: TokenizerLike,
    hard_limit: int,
) -> list[DocumentChunk]:
    """Build boundary-aware chunks for one candidate target."""
    if target_tokens <= 0:
        raise ValueError(
            "Target tokens must be positive."
        )

    if hard_limit <= 0:
        raise ValueError(
            "Hard token limit must be positive."
        )

    if target_tokens > hard_limit:
        raise ValueError(
            "Target tokens cannot exceed the hard token limit."
        )

    chunks: list[DocumentChunk] = []

    for section_index, section in enumerate(sections):
        if not section:
            continue

        heading_units = leading_heading_units(section)

        body_units = tuple(
            section[len(heading_units):]
        )

        # A heading-only section still needs to be represented.
        if not body_units:
            heading_text = section_text(heading_units)

            heading_only_chunk = make_chunk(
                document_id=document_id,
                section_index=section_index,
                chunk_index=0,
                target_tokens=target_tokens,
                heading_units=(),
                content_units=heading_units,
                tokenizer=tokenizer,
            )

            if heading_only_chunk.token_count > hard_limit:
                raise ValueError(
                    "Heading-only section exceeds the hard "
                    f"token limit: {heading_text[:80]}"
                )

            chunks.append(heading_only_chunk)
            continue

        current_units: list[DocumentUnit] = []
        chunk_index = 0

        for unit in body_units:
            unit_token_count = count_tokens(
                unit.text,
                tokenizer,
            )

            if unit_token_count > hard_limit:
                raise ValueError(
                    "A semantic unit exceeds the embedding model "
                    f"limit on page {unit.page_number}: "
                    f"{unit_token_count} > {hard_limit}"
                )

            candidate_units = [
                *current_units,
                unit,
            ]

            candidate_text = "\n".join(
                candidate_unit.text
                for candidate_unit in candidate_units
            )

            candidate_embedding_text = compose_chunk_text(
                (
                    heading.text
                    for heading in heading_units
                ),
                candidate_text,
            )

            candidate_token_count = count_tokens(
                candidate_embedding_text,
                tokenizer,
            )

            if (
                current_units
                and candidate_token_count > target_tokens
            ):
                chunk = make_chunk(
                    document_id=document_id,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    target_tokens=target_tokens,
                    heading_units=heading_units,
                    content_units=current_units,
                    tokenizer=tokenizer,
                )

                if chunk.token_count > hard_limit:
                    raise ValueError(
                        f"Chunk {chunk.chunk_id} exceeds "
                        f"the hard token limit."
                    )

                chunks.append(chunk)
                chunk_index += 1

                current_units = [unit]

                single_text = compose_chunk_text(
                    (
                        heading.text
                        for heading in heading_units
                    ),
                    unit.text,
                )

                single_token_count = count_tokens(
                    single_text,
                    tokenizer,
                )

                if single_token_count > hard_limit:
                    raise ValueError(
                        "Heading context plus semantic unit exceeds "
                        f"the hard token limit on page "
                        f"{unit.page_number}: "
                        f"{single_token_count} > {hard_limit}"
                    )

                continue

            if candidate_token_count > hard_limit:
                raise ValueError(
                    "Unable to construct a chunk below the hard "
                    f"token limit for section {section_index}."
                )

            current_units.append(unit)

        if current_units:
            chunk = make_chunk(
                document_id=document_id,
                section_index=section_index,
                chunk_index=chunk_index,
                target_tokens=target_tokens,
                heading_units=heading_units,
                content_units=current_units,
                tokenizer=tokenizer,
            )

            if chunk.token_count > hard_limit:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} exceeds "
                    f"the hard token limit."
                )

            chunks.append(chunk)

    return chunks