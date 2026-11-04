"""Upload and extract unstructured HAZOP reports into the typed graph."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_ai_provider_factory, get_graph_repository
from app.application.use_cases.extract_and_store_report import ExtractAndStoreReport
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.services.safety_validator import HazopSafetyValidator
from app.infrastructure.extraction.document_text_extractor import extract_document_text
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory

router = APIRouter(tags=["report-extraction"])
MAX_REPORT_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".log"}


@router.post(
    "/reports/extract",
    status_code=201,
    summary="Extract typed HAZOP knowledge from a PDF or text report and persist it",
)
async def extract_report(
    file: Annotated[
        UploadFile,
        File(description="PDF or UTF-8 .txt, .md, .log HAZOP report"),
    ],
    dataset_id: Annotated[str, Form(min_length=1, max_length=80)],
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    ai_factory: Annotated[AIProviderFactory | None, Depends(get_ai_provider_factory)],
    name: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF, UTF-8 .txt, .md or .log report.",
        )

    try:
        raw = await file.read(MAX_REPORT_BYTES + 1)
    finally:
        await file.close()
    if len(raw) > MAX_REPORT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Report exceeds the 20 MB upload limit.",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded report is empty.")

    selected_id = dataset_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", selected_id):
        raise HTTPException(
            status_code=422,
            detail="dataset_id must contain only letters, numbers, dot, underscore and hyphen.",
        )
    if ai_factory is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is required. Add it to the repository-root .env "
                "and restart FastAPI."
            ),
        )

    try:
        document = extract_document_text(filename, raw)
        extractor = ai_factory.create_report_extractor()
        use_case = ExtractAndStoreReport(
            extractor, repository, HazopSafetyValidator()
        )
        title = name or filename or selected_id
        metadata, validation, records = use_case.execute(
            document.text, title=title, dataset_id=selected_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "extracted_and_imported",
        "persisted": True,
        "dataset": metadata.to_dict(),
        "document": {
            "filename": filename,
            "format": suffix.lstrip("."),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "page_count": document.page_count,
            "empty_pages": list(document.empty_pages),
        },
        "scenario_count": len(records),
        "relationship_count": metadata.relationship_count,
        "validation": validation.to_dict(),
        "notice": (
            "All extracted relationships have origin llm_extracted and remain "
            "unreviewed. Exact source quotes are stored as evidence. Page markers "
            "are retained in source locations where extracted by the model. "
            "This is not expert validation. PDF text extraction does not perform OCR."
        ),
    }
