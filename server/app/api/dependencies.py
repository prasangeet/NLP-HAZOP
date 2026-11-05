"""FastAPI dependencies for graph storage and AI providers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, Request

from app.domain.repositories.graph_repository import DatasetRepository
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory
from app.infrastructure.factories.gemini_ai_provider_factory import (
    GeminiAIProviderFactory,
)


SERVER_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SERVER_ROOT.parent

# Load general configuration first, then server-specific configuration.
# Existing process environment variables always take precedence.
load_dotenv(REPOSITORY_ROOT / ".env")
load_dotenv(SERVER_ROOT / ".env")


def get_graph_repository(request: Request) -> DatasetRepository:
    """Return the configured repository or a service-unavailable error."""

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


def get_ai_provider_factory() -> AIProviderFactory | None:
    """Return the Gemini provider factory when an API key is configured."""

    if not os.getenv("GEMINI_API_KEY"):
        return None

    return GeminiAIProviderFactory()
