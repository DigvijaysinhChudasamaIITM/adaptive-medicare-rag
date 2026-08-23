from collections.abc import Sequence

from app.models.generation import GeneratedAnswer
from app.models.grounding import SourceEvidence
from app.rag.prompting import INSUFFICIENT_EVIDENCE_ANSWER
from app.rag.vector_store import SearchHit

DEFAULT_SNIPPET_MAX_CHARS = 420


class CitationIntegrityError(ValueError):
    """Raised when model citations violate the retrieved evidence boundary."""


def validate_citations(
    generated: GeneratedAnswer,
    retrieved_hits: Sequence[SearchHit],
) -> tuple[SearchHit, ...]:
    """Validate LLM citation IDs against retrieved evidence.

    Citation order follows the LLM output. Duplicate citations are
    deterministically collapsed to their first occurrence.
    """

    hits_by_id: dict[str, SearchHit] = {}

    for hit in retrieved_hits:
        chunk_id = hit.chunk.chunk_id

        if chunk_id in hits_by_id:
            raise CitationIntegrityError(
                "Retrieved evidence contains duplicate chunk IDs."
            )

        hits_by_id[chunk_id] = hit

    is_abstention = (
        generated.answer
        == INSUFFICIENT_EVIDENCE_ANSWER
    )

    if is_abstention:
        if generated.citations:
            raise CitationIntegrityError(
                "An abstention response must not contain citations."
            )

        return ()

    if not generated.citations:
        raise CitationIntegrityError(
            "A substantive answer must contain at least one citation."
        )

    validated: list[SearchHit] = []
    seen_ids: set[str] = set()

    for citation_id in generated.citations:
        if citation_id in seen_ids:
            continue

        hit = hits_by_id.get(citation_id)

        if hit is None:
            raise CitationIntegrityError(
                "Citation ID is not present in the retrieved evidence set: "
                f"{citation_id}"
            )

        seen_ids.add(citation_id)
        validated.append(hit)

    return tuple(validated)


def build_source_evidence(
    cited_hits: Sequence[SearchHit],
    *,
    snippet_max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
) -> tuple[SourceEvidence, ...]:
    """Build trusted source objects from validated retrieval hits."""

    if snippet_max_chars < 20:
        raise ValueError(
            "snippet_max_chars must be at least 20."
        )

    sources: list[SourceEvidence] = []

    for hit in cited_hits:
        page_numbers = list(
            hit.chunk.page_numbers
        )

        sources.append(
            SourceEvidence(
                chunk_id=hit.chunk.chunk_id,
                page_numbers=page_numbers,
                page_start=hit.chunk.page_start,
                page_end=hit.chunk.page_end,
                page_reference=_build_page_reference(
                    page_numbers
                ),
                snippet=_build_snippet(
                    hit.chunk.text,
                    max_chars=snippet_max_chars,
                ),
                retrieval_score=float(
                    hit.score
                ),
                retrieval_rank=hit.rank,
            )
        )

    return tuple(sources)


def _build_snippet(
    text: str,
    *,
    max_chars: int,
) -> str:
    """Create a deterministic bounded excerpt from trusted chunk text."""

    normalized = " ".join(
        text.split()
    )

    if not normalized:
        raise ValueError(
            "Cannot build a snippet from empty chunk text."
        )

    if len(normalized) <= max_chars:
        return normalized

    cutoff = max_chars - 1
    candidate = normalized[:cutoff].rstrip()

    boundary = candidate.rfind(" ")

    if boundary >= int(cutoff * 0.60):
        candidate = candidate[:boundary].rstrip()

    return candidate + "…"


def _build_page_reference(
    page_numbers: list[int],
) -> str:
    """Create a deterministic physical-PDF page reference."""

    if not page_numbers:
        raise ValueError(
            "At least one page number is required."
        )

    if len(page_numbers) == 1:
        return f"PDF page {page_numbers[0]}"

    expected_contiguous = list(
        range(
            page_numbers[0],
            page_numbers[-1] + 1,
        )
    )

    if page_numbers == expected_contiguous:
        return (
            f"PDF pages "
            f"{page_numbers[0]}–{page_numbers[-1]}"
        )

    joined = ", ".join(
        str(page)
        for page in page_numbers
    )

    return f"PDF pages {joined}"