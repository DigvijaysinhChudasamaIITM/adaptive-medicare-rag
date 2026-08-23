from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroundingEvidence(BaseModel):
    """Trusted retrieval evidence supplied to the generation layer."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("chunk_id", "text")
    @classmethod
    def strip_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty or whitespace only.")
        return cleaned


class GeneratedAnswer(BaseModel):
    """Minimal structured output that the LLM is allowed to author."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str]

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Answer cannot be empty or whitespace only.")
        return cleaned

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []

        for value in values:
            citation = value.strip()
            if not citation:
                raise ValueError("Citation IDs cannot be blank.")
            cleaned.append(citation)

        return cleaned