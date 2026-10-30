"""Upload and extract unstructured HAZOP reports into the typed graph."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from app.api.dependencies import get_ai_provider_factory, get_graph_repository
from app.application.use_cases.extract_and_store_report import ExtractAndStoreReport
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.services.safety_validator import HazopSafetyValidator
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory

router = APIRouter(tags=["report-extraction"])
MAX_REPORT_BYTES = 2 * 1024 * 1024

@router.post("/reports/extract", status_code=201, summary="Extract typed HAZOP knowledge from a text report and persist it")
async def extract_report(
    file: Annotated[UploadFile, File(description="UTF-8 .txt or .md HAZOP report")],
    dataset_id: Annotated[str, Form(min_length=1, max_length=80)],
    name: Annotated[str | None, Form()] = None,
    repository: DatasetRepository = Depends(get_graph_repository),
    ai_factory: AIProviderFactory | None = Depends(get_ai_provider_factory),
) -> dict[str, object]:
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.casefold() not in {".txt", ".md", ".log"}:
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a UTF-8 .txt, .md or .log report. PDF ingestion is not enabled yet.")
    raw = await file.read(MAX_REPORT_BYTES + 1)
    await file.close()
    if len(raw) > MAX_REPORT_BYTES:
        raise HTTPException(status_code=413, detail="Report exceeds the 2 MB upload limit.")
    try:
        source_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Report must be UTF-8 text.") from exc
    selected_id = dataset_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", selected_id):
        raise HTTPException(status_code=422, detail="dataset_id must contain only letters, numbers, dot, underscore and hyphen.")
    if ai_factory is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is required. Add it to the repository-root .env and restart FastAPI.")
    try:
        extractor = ai_factory.create_report_extractor()
        use_case = ExtractAndStoreReport(extractor, repository, HazopSafetyValidator())
        metadata, validation, records = use_case.execute(source_text, title=(name or filename or selected_id), dataset_id=selected_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "extracted_and_imported",
        "persisted": True,
        "dataset": metadata.to_dict(),
        "scenario_count": len(records),
        "relationship_count": metadata.relationship_count,
        "validation": validation.to_dict(),
        "notice": "All extracted relationships have origin llm_extracted and remain unreviewed. Exact source quotes are stored as evidence; this is not expert validation.",
    }
