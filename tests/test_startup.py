from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import (
    RuntimeResources,
    build_runtime,
    create_app,
)
from app.rag.manifest import (
    ArtifactCompatibilityError,
    CompatibilityResult,
)


class FakeCloser:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeService:
    async def answer(self, query: str):
        raise AssertionError(
            "Query should not be called."
        )


def make_settings() -> Settings:
    return Settings(
        _env_file=None,
        app_name="Medicare RAG API",
        openrouter_api_key="test-key",
        llm_model="test-model",
        embedding_model="test-embedding",
        top_k=10,
        final_top_k=4,
    )


def make_compatibility() -> CompatibilityResult:
    return CompatibilityResult(
        document_sha256="abc",
        embedding_model="test-embedding",
        embedding_dimension=384,
        strategy_id="target_416",
        target_tokens=416,
        chunk_count=481,
        index_type="IndexFlatIP",
        relevance_threshold=0.7607258856296539,
    )


def test_lifespan_builds_and_closes_runtime() -> None:
    closer = FakeCloser()
    called = False

    def runtime_builder(
        settings: Settings,
    ) -> RuntimeResources:
        nonlocal called
        called = True

        return RuntimeResources(
            rag_service=FakeService(),  # type: ignore[arg-type]
            openrouter_client=closer,  # type: ignore[arg-type]
            compatibility=make_compatibility(),
        )

    application = create_app(
        settings=make_settings(),
        runtime_builder=runtime_builder,
    )

    with TestClient(
        application
    ) as client:
        response = client.get(
            "/health"
        )

        assert response.status_code == 200
        assert called is True
        assert closer.closed is False

    assert closer.closed is True


def test_startup_fails_closed_on_artifact_error() -> None:
    def failing_builder(
        settings: Settings,
    ) -> RuntimeResources:
        raise ArtifactCompatibilityError(
            "manifest mismatch"
        )

    application = create_app(
        settings=make_settings(),
        runtime_builder=failing_builder,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="manifest mismatch",
    ):
        with TestClient(
            application
        ):
            pass


def test_build_runtime_validates_expected_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_validate_runtime_compatibility(
        *,
        manifest_path: Path,
        pdf_path: Path,
        selected_strategy_path: Path,
        relevance_calibration_path: Path,
        index_directory: Path,
        expected_embedding_model: str,
    ) -> CompatibilityResult:
        calls.update(
            {
                "manifest_path": manifest_path,
                "pdf_path": pdf_path,
                "selected_strategy_path": (
                    selected_strategy_path
                ),
                "relevance_calibration_path": (
                    relevance_calibration_path
                ),
                "index_directory": index_directory,
                "embedding_model": (
                    expected_embedding_model
                ),
            }
        )

        raise ArtifactCompatibilityError(
            "stop after validation"
        )

    monkeypatch.setattr(
        main_module,
        "validate_runtime_compatibility",
        fake_validate_runtime_compatibility,
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="stop after validation",
    ):
        build_runtime(
            make_settings()
        )

    assert calls == {
        "manifest_path": Path(
            "artifacts/manifest.json"
        ),
        "pdf_path": Path(
            "data/medicare.pdf"
        ),
        "selected_strategy_path": Path(
            "artifacts/selected_strategy.json"
        ),
        "relevance_calibration_path": Path(
            "artifacts/relevance_calibration.json"
        ),
        "index_directory": Path(
            "artifacts/indexes/selected"
        ),
        "embedding_model": "test-embedding",
    }