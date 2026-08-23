from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from app.clients.openrouter import (
    OpenRouterConfigurationError,
    OpenRouterProviderError,
    OpenRouterResponseError,
)
from app.models.api import QueryRequest
from app.models.grounding import GroundedAnswer
from app.rag.citations import CitationIntegrityError
from app.rag.confidence import EvidenceConfidenceError
from app.rag.service import (
    RAGService,
    RAGServiceError,
    RetrievalServiceError,
)

router = APIRouter()


def _get_rag_service(
    request: Request,
) -> RAGService:
    service = getattr(
        request.app.state,
        "rag_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not ready.",
        )

    return service


@router.get("/health")
async def health(
    request: Request,
) -> dict[str, str]:
    """Return basic API service health."""

    return {
        "status": "healthy",
        "service": request.app.title,
    }


@router.post(
    "/query",
    response_model=GroundedAnswer,
)
async def query(
    payload: QueryRequest,
    request: Request,
) -> GroundedAnswer:
    """Answer a question using the validated Medicare RAG pipeline."""

    service = _get_rag_service(
        request
    )

    try:
        return await service.answer(
            payload.query
        )

    except RetrievalServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is unavailable.",
        ) from exc

    except OpenRouterConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation service is not configured.",
        ) from exc

    except OpenRouterProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generation provider is unavailable.",
        ) from exc

    except OpenRouterResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Generation provider returned an unusable response.",
        ) from exc

    except CitationIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Generated answer failed citation integrity validation.",
        ) from exc

    except EvidenceConfidenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence confidence could not be computed.",
        ) from exc

    except RAGServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grounded response construction failed.",
        ) from exc