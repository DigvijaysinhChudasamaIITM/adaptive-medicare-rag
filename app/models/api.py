from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_CHARS = 2000


class QueryRequest(BaseModel):
    """Validated user query accepted by the RAG API."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
    )