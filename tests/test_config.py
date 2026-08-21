import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_retrieval_configuration_is_valid() -> None:
    settings = Settings()

    assert settings.top_k == 10
    assert settings.final_top_k == 4


def test_final_top_k_cannot_exceed_top_k() -> None:
    with pytest.raises(ValidationError):
        Settings(top_k=3, final_top_k=4)