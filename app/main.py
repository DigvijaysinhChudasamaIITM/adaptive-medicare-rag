from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.clients.openrouter import OpenRouterClient
from app.config import Settings, get_settings
from app.rag.manifest import (
    CompatibilityResult,
    validate_runtime_compatibility,
)
from app.rag.relevance import RelevanceGate
from app.rag.retrieval import Retriever
from app.rag.service import RAGService

PDF_PATH = Path("data/medicare.pdf")
MANIFEST_PATH = Path("artifacts/manifest.json")
SELECTED_STRATEGY_PATH = Path(
    "artifacts/selected_strategy.json"
)
RELEVANCE_CALIBRATION_PATH = Path(
    "artifacts/relevance_calibration.json"
)
INDEX_DIRECTORY = Path(
    "artifacts/indexes/selected"
)


@dataclass(slots=True)
class RuntimeResources:
    """Runtime objects owned by the FastAPI application lifespan."""

    rag_service: RAGService
    openrouter_client: OpenRouterClient
    compatibility: CompatibilityResult

    async def aclose(self) -> None:
        """Release runtime network resources."""

        await self.openrouter_client.aclose()


RuntimeBuilder = Callable[
    [Settings],
    RuntimeResources,
]


def build_runtime(
    settings: Settings,
) -> RuntimeResources:
    """Validate artifacts and construct production RAG dependencies."""

    compatibility = (
        validate_runtime_compatibility(
            manifest_path=MANIFEST_PATH,
            pdf_path=PDF_PATH,
            selected_strategy_path=(
                SELECTED_STRATEGY_PATH
            ),
            relevance_calibration_path=(
                RELEVANCE_CALIBRATION_PATH
            ),
            index_directory=INDEX_DIRECTORY,
            expected_embedding_model=(
                settings.embedding_model
            ),
        )
    )

    retriever = Retriever.load(
        index_directory=INDEX_DIRECTORY,
        embedding_model=(
            settings.embedding_model
        ),
        default_top_k=settings.top_k,
    )

    relevance_gate = RelevanceGate.load(
        RELEVANCE_CALIBRATION_PATH
    )

    openrouter_client = OpenRouterClient(
        settings=settings
    )

    rag_service = RAGService(
        retriever=retriever,
        relevance_gate=relevance_gate,
        generator=openrouter_client,
        top_k=settings.top_k,
        final_top_k=settings.final_top_k,
    )

    return RuntimeResources(
        rag_service=rag_service,
        openrouter_client=openrouter_client,
        compatibility=compatibility,
    )


def create_app(
    *,
    settings: Settings | None = None,
    runtime_builder: RuntimeBuilder | None = None,
) -> FastAPI:
    """Create the FastAPI application and lifespan."""

    resolved_settings = (
        settings
        if settings is not None
        else get_settings()
    )

    resolved_runtime_builder = (
        runtime_builder
        if runtime_builder is not None
        else build_runtime
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        runtime = resolved_runtime_builder(
            resolved_settings
        )

        application.state.rag_service = (
            runtime.rag_service
        )
        application.state.compatibility = (
            runtime.compatibility
        )

        try:
            yield
        finally:
            await runtime.aclose()

    application = FastAPI(
        title=resolved_settings.app_name,
        version="0.1.0",
        description=(
            "RAG API for the supplied Medicare handbook."
        ),
        lifespan=lifespan,
    )

    application.include_router(
        router
    )

    return application


app = create_app()