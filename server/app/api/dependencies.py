"""FastAPI dependencies for configured graph storage."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.domain.repositories.graph_repository import DatasetRepository


def get_graph_repository(request: Request) -> DatasetRepository:
    """Return the configured repository or a clear service-unavailable error."""
    repository = getattr(request.app.state, "graph_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph storage is not configured. Set NEO4J_HTTP_URI, "
                "NEO4J_USERNAME, and NEO4J_PASSWORD to enable dataset APIs."
            ),
        )
    return repository


def get_ai_provider_factory():
    """Return the configured AI Abstract Factory, or None when no key is set."""
    import os

    if not os.getenv("GROQ_API_KEY"):
        return None
    from app.infrastructure.factories.groq_ai_provider_factory import (
        GroqAIProviderFactory,
    )

    return GroqAIProviderFactory()
