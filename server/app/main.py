"""FastAPI application entry point for NLP-HAZOP."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.imports import router as imports_router


def create_app() -> FastAPI:
    """Create the API application.

    CORS defaults cover the Vite development server and common Tauri origins.
    Additional origins can be supplied through NLP_HAZOP_CORS_ORIGINS as a
    comma-separated environment variable.
    """
    app = FastAPI(
        title="NLP-HAZOP API",
        description=(
            "Local API for importing and inspecting structured HAZOP data. "
            "Uploaded CSVs are previewed but are not persisted yet."
        ),
        version="0.1.0",
    )

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
    return app


app = create_app()
