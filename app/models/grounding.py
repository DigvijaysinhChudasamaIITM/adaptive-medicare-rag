from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class SourceEvidence(BaseModel):
    """Backend-owned trusted source metadata for one validated citation."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    page_numbers: list[int] = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    page_reference: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    retrieval_score: float = Field(ge=-1.0, le=1.0)
    retrieval_rank: int = Field(ge=1)

    @field_validator(
        "chunk_id",
        "page_reference",
        "snippet",
    )
    @classmethod
    def strip_non_empty_text(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Source text fields cannot be empty or whitespace only."
            )

        return cleaned

    @model_validator(mode="after")
    def validate_page_provenance(
        self,
    ) -> "SourceEvidence":
        if any(page <= 0 for page in self.page_numbers):
            raise ValueError(
                "Page numbers must be positive."
            )

        if (
            self.page_numbers
            != sorted(set(self.page_numbers))
        ):
            raise ValueError(
                "Page numbers must be sorted and unique."
            )

        if self.page_start != self.page_numbers[0]:
            raise ValueError(
                "page_start must match the first page number."
            )

        if self.page_end != self.page_numbers[-1]:
            raise ValueError(
                "page_end must match the last page number."
            )

        return self


class GroundedAnswer(BaseModel):
    """Backend-enriched grounded answer ready for API orchestration."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[SourceEvidence]

    @field_validator("answer")
    @classmethod
    def strip_answer(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Answer cannot be empty or whitespace only."
            )

        return cleaned