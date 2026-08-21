from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Medicare RAG API"
    environment: str = "development"

    openrouter_api_key: str | None = None
    llm_model: str | None = None
    llm_fallback_model: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    top_k: int = Field(default=10, ge=1, le=50)
    final_top_k: int = Field(default=4, ge=1, le=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "Settings":
        if self.final_top_k > self.top_k:
            raise ValueError("FINAL_TOP_K cannot be greater than TOP_K.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()