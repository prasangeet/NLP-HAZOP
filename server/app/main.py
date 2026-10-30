"""FastAPI application entry point for NLP-HAZOP."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.datasets import router as datasets_router
from app.api.routes.imports import router as imports_router
from app.api.routes.recovery import router as recovery_router
from app.api.routes.report_extraction import router as report_extraction_router
from app.infrastructure.graph.neo4j_graph_repository import Neo4jGraphRepository


def _configured_repository() -> Neo4jGraphRepository | None:
    """Create the durable repository when complete Neo4j settings are present."""
    uri = os.getenv("NEO4J_HTTP_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    configured = (uri, username, password)
    if not any(configured):
        return None
    if not all(configured):
        raise RuntimeError(
            "Configure NEO4J_HTTP_URI, NEO4J_USERNAME, and NEO4J_PASSWORD together."
        )
    timeout = float(os.getenv("NEO4J_TIMEOUT_SECONDS", "15"))
    return Neo4jGraphRepository(
        http_uri=uri,
        username=username,
        password=password,
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        timeout=timeout,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release the configured Neo4j HTTP client during application shutdown."""
    yield
    repository = getattr(app.state, "graph_repository", None)
    if repository is not None and hasattr(repository, "close"):
        repository.close()


def create_app() -> FastAPI:
    """Create the API application.

    Preview-only CSV validation works without Neo4j. Confirmed imports,
    dataset/graph queries, and recovery require configured persistent storage.
    """
    app = FastAPI(
        title="NLP-HAZOP API",
        description=(
            "API for importing, inspecting, and recovering missing links in "
            "structured HAZOP datasets. Preview does not persist data; "
            "confirmed imports require Neo4j."
        ),
        version="0.2.0",
        lifespan=_lifespan,
    )
    app.state.graph_repository = _configured_repository()

    default_origins = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:1420",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    }
    configured_origins = {
        origin.strip()
        for origin in os.getenv("NLP_HAZOP_CORS_ORIGINS", "").split(",")
        if origin.strip()
    }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(default_origins | configured_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Return a lightweight process health check."""
        return {"status": "ok", "service": "nlp-hazop-api"}

    app.include_router(imports_router, prefix="/api/v1")
    app.include_router(datasets_router, prefix="/api/v1")
    app.include_router(recovery_router, prefix="/api/v1")
    app.include_router(report_extraction_router, prefix="/api/v1")
    return app


app = create_app()
