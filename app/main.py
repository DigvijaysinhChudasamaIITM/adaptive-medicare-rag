from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="RAG API for the supplied Medicare handbook.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return basic API service health."""
    return {
        "status": "healthy",
        "service": settings.app_name,
    }