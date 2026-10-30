"""Dataset import and preview endpoints."""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.domain.entities.hazop_record import (
    RecordType,
    ValidationStatus,
)
from app.infrastructure.datasets.hazop_csv_loader import HazopCsvLoader

router = APIRouter(tags=["datasets"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_PREVIEW_LIMIT = 100
MAX_PREVIEW_LIMIT = 500

SCENARIO_ID_HEADERS = {"scenario_id", "scenario id"}
DEVIATION_HEADERS = {"deviation"}


class CsvPreviewResponse(BaseModel):
    """Normalized preview returned after a CSV upload."""

    status: str
    filename: str
    dataset_id: str
    storage: str
    recognized_columns: dict[str, str]
    summary: dict[str, Any]
    warnings: list[str]
    preview_limit: int
    preview_truncated: bool
    records: list[dict[str, Any]]


def _safe_filename(filename: str | None) -> str:
    """Return a display-safe basename without directory components."""
    raw = (filename or "").replace("\\", "/")
    return Path(raw).name


def _dataset_id(filename: str) -> str:
    stem = Path(filename).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    return (slug or "uploaded-dataset")[:80]


def _find_header(headers: list[str], accepted: set[str]) -> str | None:
    for header in headers:
        if header.strip().casefold() in accepted:
            return header
    return None


@router.post(
    "/datasets/preview",
    response_model=CsvPreviewResponse,
    summary="Upload and preview a HAZOP CSV",
)
async def preview_csv(
    file: UploadFile = File(..., description="UTF-8 CSV HAZOP worksheet"),
    preview_limit: int = Query(
        DEFAULT_PREVIEW_LIMIT,
        ge=1,
        le=MAX_PREVIEW_LIMIT,
        description="Maximum number of normalized records returned in the preview",
    ),
) -> CsvPreviewResponse:
    """Validate and normalize a CSV without persisting or modifying data."""
    filename = _safe_filename(file.filename)
    if not filename or Path(filename).suffix.casefold() != ".csv":
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Upload a .csv file. Excel workbooks are not supported yet.",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")

    try:
        csv_text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="CSV must use UTF-8 encoding (a UTF-8 BOM is supported).",
        ) from exc

    try:
        reader = csv.DictReader(io.StringIO(csv_text, newline=""))
        headers = reader.fieldnames
        if not headers or not any(header.strip() for header in headers):
            raise HTTPException(
                status_code=422,
                detail="CSV must contain a non-empty header row.",
            )

        normalized_headers = [header.strip().casefold() for header in headers]
        if any(not header for header in normalized_headers):
            raise HTTPException(
                status_code=422,
                detail="CSV contains an empty column name.",
            )
        if len(normalized_headers) != len(set(normalized_headers)):
            raise HTTPException(
                status_code=422,
                detail="CSV contains duplicate column names.",
            )

        scenario_header = _find_header(headers, SCENARIO_ID_HEADERS)
        deviation_header = _find_header(headers, DEVIATION_HEADERS)
        if scenario_header is None or deviation_header is None:
            missing = []
            if scenario_header is None:
                missing.append("Scenario_ID (or Scenario ID)")
            if deviation_header is None:
                missing.append("Deviation")
            raise HTTPException(
                status_code=422,
                detail="Missing required column(s): " + ", ".join(missing),
            )

        rows = list(reader)
    except csv.Error as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse CSV: {exc}",
        ) from exc

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="CSV contains a header but no data rows.",
        )

    dataset_id = _dataset_id(filename)
    try:
        records = HazopCsvLoader().load_stream(
            io.StringIO(csv_text, newline=""),
            dataset_id=dataset_id,
        )
    except (csv.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not normalize CSV records: {exc}",
        ) from exc

    if not records:
        raise HTTPException(
            status_code=422,
            detail=(
                "No usable HAZOP records were found. Check that data rows "
                "contain a Scenario_ID value."
            ),
        )

    scenario_ids = [record.scenario_id.casefold() for record in records]
    duplicate_count = sum(
        count - 1 for count in Counter(scenario_ids).values() if count > 1
    )
    skipped_rows = [
        index + 2
        for index, row in enumerate(rows)
        if not str(row.get(scenario_header) or "").strip()
    ]
    malformed_rows = [
        index + 2
        for index, row in enumerate(rows)
        if None in row or any(value is None for value in row.values())
    ]

    record_type_counts = Counter(record.record_type.value for record in records)
    validation_counts = Counter(
        record.validation_status.value for record in records
    )
    relationship_count = sum(len(record.causal_relations) for record in records)

    warnings = [
        (
            "Relationships are generated from the current CSV-to-domain mapping. "
            "Co-listed fields are associations, not independently verified "
            "causal assertions."
        ),
        (
            "Preview only: uploaded records have not been saved to a database "
            "or retained after this request."
        ),
    ]
    if skipped_rows:
        shown = ", ".join(str(number) for number in skipped_rows[:20])
        suffix = " ..." if len(skipped_rows) > 20 else ""
        warnings.append(
            f"Skipped {len(skipped_rows)} row(s) without a Scenario_ID "
            f"(CSV line(s): {shown}{suffix})."
        )
    if malformed_rows:
        shown = ", ".join(str(number) for number in malformed_rows[:20])
        suffix = " ..." if len(malformed_rows) > 20 else ""
        warnings.append(
            f"{len(malformed_rows)} row(s) have a different number of cells "
            f"than the header (CSV line(s): {shown}{suffix})."
        )
    if duplicate_count:
        warnings.append(
            f"Found {duplicate_count} duplicate Scenario_ID occurrence(s). "
            "Review these before importing into persistent storage."
        )
    if any(
        record.record_type == RecordType.SYNTHETIC
        or record.validation_status == ValidationStatus.UNREVIEWED
        for record in records
    ):
        warnings.append(
            "The returned provenance and validation status are preserved; "
            "the API does not promote synthetic or unreviewed data to "
            "expert-validated knowledge."
        )

    preview_records = records[:preview_limit]
    return CsvPreviewResponse(
        status="preview_ready",
        filename=filename,
        dataset_id=dataset_id,
        storage="not_persisted",
        recognized_columns={
            "scenario_id": scenario_header,
            "deviation": deviation_header,
        },
        summary={
            "rows_read": len(rows),
            "records_loaded": len(records),
            "rows_skipped": len(skipped_rows),
            "relationships_generated": relationship_count,
            "record_type_counts": dict(record_type_counts),
            "validation_status_counts": dict(validation_counts),
        },
        warnings=warnings,
        preview_limit=preview_limit,
        preview_truncated=len(records) > preview_limit,
        records=[record.to_dict() for record in preview_records],
    )
