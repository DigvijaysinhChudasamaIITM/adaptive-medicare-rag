from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.clients.openrouter import (
    OpenRouterProviderError,
    OpenRouterResponseError,
)
from app.config import Settings
from app.main import (
    RuntimeResources,
    create_app,
)
from app.models.grounding import (
    GroundedAnswer,
    SourceEvidence,
)
from app.rag.citations import CitationIntegrityError


class FakeCloser:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class FakeCompatibility:
    document_sha256: str = "test"


class FakeService:
    def __init__(
        self,
        *,
        result: GroundedAnswer | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.queries: list[str] = []

    async def answer(
        self,
        query: str,
    ) -> GroundedAnswer:
        self.queries.append(query)

        if self.error is not None:
            raise self.error

        assert self.result is not None
        return self.result


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


def make_app(
    service: FakeService,
):
    closer = FakeCloser()

    def runtime_builder(
        settings: Settings,
    ) -> RuntimeResources:
        return RuntimeResources(
            rag_service=service,  # type: ignore[arg-type]
            openrouter_client=closer,  # type: ignore[arg-type]
            compatibility=FakeCompatibility(),  # type: ignore[arg-type]
        )

    return (
        create_app(
            settings=make_settings(),
            runtime_builder=runtime_builder,
        ),
        closer,
    )


def test_query_returns_grounded_json() -> None:
    result = GroundedAnswer(
        answer="Medicare covers the visit.",
        confidence_score=0.78,
        sources=[
            SourceEvidence(
                chunk_id="chunk-1",
                page_numbers=[54, 55],
                page_start=54,
                page_end=55,
                page_reference="PDF pages 54–55",
                snippet="Trusted source snippet.",
                retrieval_score=0.87,
                retrieval_rank=1,
            )
        ],
    )

    application, closer = make_app(
        FakeService(result=result)
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={
                "query": (
                    "Does Medicare cover "
                    "a Wellness visit?"
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == result.model_dump()
    assert closer.closed is True


def test_blank_query_returns_422() -> None:
    result = GroundedAnswer(
        answer="Unused.",
        confidence_score=0.0,
        sources=[],
    )

    application, _ = make_app(
        FakeService(result=result)
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={"query": "   "},
        )

    assert response.status_code == 422


def test_query_over_limit_returns_422() -> None:
    result = GroundedAnswer(
        answer="Unused.",
        confidence_score=0.0,
        sources=[],
    )

    application, _ = make_app(
        FakeService(result=result)
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={"query": "x" * 2001},
        )

    assert response.status_code == 422


def test_extra_request_field_returns_422() -> None:
    result = GroundedAnswer(
        answer="Unused.",
        confidence_score=0.0,
        sources=[],
    )

    application, _ = make_app(
        FakeService(result=result)
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={
                "query": "Medicare question",
                "extra": "not allowed",
            },
        )

    assert response.status_code == 422


def test_provider_failure_returns_503() -> None:
    service = FakeService(
        error=OpenRouterProviderError(
            "provider unavailable",
            status_code=503,
            retryable=True,
            fallback_allowed=True,
        )
    )

    application, _ = make_app(
        service
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={"query": "Medicare question"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Generation provider is unavailable."
    }


def test_malformed_generation_returns_502() -> None:
    application, _ = make_app(
        FakeService(
            error=OpenRouterResponseError(
                "malformed provider response"
            )
        )
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={"query": "Medicare question"},
        )

    assert response.status_code == 502


def test_citation_integrity_failure_returns_502() -> None:
    application, _ = make_app(
        FakeService(
            error=CitationIntegrityError(
                "fake citation"
            )
        )
    )

    with TestClient(
        application
    ) as client:
        response = client.post(
            "/query",
            json={"query": "Medicare question"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Generated answer failed "
            "citation integrity validation."
        )
    }