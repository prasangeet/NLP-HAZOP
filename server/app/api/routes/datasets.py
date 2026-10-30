"""Persistent dataset import and query endpoints."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.api.dependencies import get_graph_repository
from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.hazop_record import HazopRecord
from app.domain.repositories.graph_repository import DatasetRepository
from app.infrastructure.datasets.hazop_csv_loader import (
    COLUMN_ALIASES,
    HazopCsvLoader,
)
from app.infrastructure.graph.neo4j_graph_repository import Neo4jRepositoryError

router = APIRouter(tags=["datasets"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _safe_filename(filename: str | None) -> str:
    """Return a display-safe basename without directory components."""
    raw = (filename or "").replace("\\", "/")
    return Path(raw).name


def _default_dataset_id(filename: str) -> str:
    """Derive a stable, URL-friendly dataset ID from the uploaded filename."""
    stem = Path(filename).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    return (slug or "uploaded-dataset")[:80]


def _validate_and_load_csv(
    filename: str,
    content: bytes,
    dataset_id: str,
) -> tuple[HazopRecord, ...]:
    """Validate required columns and normalize every usable CSV row."""
    if not filename or Path(filename).suffix.casefold() != ".csv":
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
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
            raise HTTPException(status_code=422, detail="CSV must contain a header row.")
        normalized = [header.strip().casefold() for header in headers]
        if any(not header for header in normalized):
            raise HTTPException(status_code=422, detail="CSV contains an empty column name.")
        if len(normalized) != len(set(normalized)):
            raise HTTPException(status_code=422, detail="CSV contains duplicate column names.")

        alias_lookup = {
            alias.casefold(): canonical
            for canonical, aliases in COLUMN_ALIASES.items()
            for alias in aliases
        }
        canonical_headers = {
            alias_lookup.get(header.strip().casefold())
            for header in headers
        }
        missing = [
            canonical
            for canonical in ("scenario_id", "deviation")
            if canonical not in canonical_headers
        ]
        if missing:
            names = {
                "scenario_id": "Scenario_ID (or Scenario ID)",
                "deviation": "Deviation",
            }
            raise HTTPException(
                status_code=422,
                detail="Missing required column(s): "
                + ", ".join(names[item] for item in missing),
            )
        rows = list(reader)
    except csv.Error as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    if not rows:
        raise HTTPException(status_code=422, detail="CSV contains a header but no data rows.")

    try:
        records = tuple(
            HazopCsvLoader().load_stream(
                io.StringIO(csv_text, newline=""),
                dataset_id=dataset_id,
            )
        )
    except (csv.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not normalize CSV records: {exc}",
        ) from exc

    if not records:
        raise HTTPException(
            status_code=422,
            detail="No usable HAZOP records were found. Check Scenario_ID values.",
        )

    scenario_ids = [record.scenario_id.casefold() for record in records]
    duplicates = [
        scenario_id
        for scenario_id, count in Counter(scenario_ids).items()
        if count > 1
    ]
    if duplicates:
        raise HTTPException(
            status_code=422,
            detail=(
                "Each Scenario_ID must be unique within a dataset. "
                f"Duplicate ID(s): {', '.join(duplicates[:20])}"
            ),
        )
    return records


@router.post(
    "/datasets/import",
    summary="Persist a validated HAZOP CSV dataset",
    status_code=201,
)
async def import_dataset(
    file: Annotated[UploadFile, File(description="UTF-8 HAZOP CSV worksheet")],
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    dataset_id: Annotated[str | None, Form()] = None,
    name: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    """Persist a full CSV dataset after explicit user confirmation."""
    filename = _safe_filename(file.filename)
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    selected_id = (dataset_id or _default_dataset_id(filename)).strip()
    if not selected_id or len(selected_id) > 80 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", selected_id
    ):
        raise HTTPException(
            status_code=422,
            detail="dataset_id must be 1-80 characters: letters, numbers, dot, underscore, hyphen.",
        )
    if not repository.is_persistent:
        raise HTTPException(
            status_code=503,
            detail="Dataset import requires persistent Neo4j storage; in-memory storage is not durable.",
        )
    if repository.get_dataset(selected_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset '{selected_id}' already exists. Choose a new dataset_id.",
        )

    records = _validate_and_load_csv(filename, content, selected_id)
    metadata = DatasetMetadata(
        dataset_id=selected_id,
        name=(name or Path(filename).stem).strip() or selected_id,
        filename=filename,
        source_sha256=hashlib.sha256(content).hexdigest(),
        record_count=len(records),
        relationship_count=sum(len(record.causal_relations) for record in records),
        created_at=datetime.now(UTC).isoformat(),
    )
    try:
        repository.import_dataset(metadata, records)
    except Neo4jRepositoryError as exc:
        if "ConstraintValidationFailed" in str(exc):
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{selected_id}' already exists.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Neo4j could not commit the dataset: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "status": "imported",
        "persisted": True,
        "storage": repository.storage_type,
        "dataset": metadata.to_dict(),
    }


@router.get("/datasets", summary="List persisted HAZOP datasets")
def list_datasets(
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, object]:
    datasets = repository.find_datasets(limit=limit)
    return {
        "storage": repository.storage_type,
        "count": len(datasets),
        "datasets": [item.to_dict() for item in datasets],
    }


@router.get("/datasets/{dataset_id}", summary="Get persisted dataset metadata")
def get_dataset(
    dataset_id: str,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
) -> dict[str, object]:
    dataset = repository.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"dataset": dataset.to_dict()}


@router.get(
    "/datasets/{dataset_id}/scenarios",
    summary="Query scenarios in a dataset",
)
def list_scenarios(
    dataset_id: str,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    equipment: str | None = None,
    deviation: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, object]:
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    records = repository.find_records(
        dataset_id=dataset_id,
        equipment=equipment,
        deviation=deviation,
        limit=limit,
    )
    return {
        "dataset_id": dataset_id,
        "count": len(records),
        "scenarios": [record.to_dict() for record in records],
    }


@router.get(
    "/datasets/{dataset_id}/scenarios/{scenario_id}",
    summary="Get one HAZOP scenario",
)
def get_scenario(
    dataset_id: str,
    scenario_id: str,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
) -> dict[str, object]:
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    record = repository.get_record(scenario_id, dataset_id=dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Scenario not found.")
    return {"scenario": record.to_dict()}


@router.get(
    "/datasets/{dataset_id}/relationships",
    summary="Query typed HAZOP relationships",
)
def list_relationships(
    dataset_id: str,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> dict[str, object]:
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    relations = repository.find_relations(
        dataset_id=dataset_id,
        subject=subject,
        predicate=predicate,
        object=object,
        limit=limit,
    )
    return {
        "dataset_id": dataset_id,
        "count": len(relations),
        "relationships": [
            {
                "subject": item.subject,
                "predicate": item.predicate,
                "object": item.object,
                "origin": item.origin.value,
                "evidence": item.evidence,
                "confidence": item.confidence,
                "source_id": item.source_id,
                "is_expert_verified": item.is_expert_verified,
                "dataset_id": item.dataset_id,
            }
            for item in relations
        ],
    }


@router.get(
    "/datasets/{dataset_id}/graph",
    summary="Return graph nodes and edges for visualization",
)
def get_graph(
    dataset_id: str,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    limit: int = Query(1000, ge=1, le=5000),
) -> dict[str, object]:
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    records = repository.find_records(dataset_id=dataset_id, limit=limit)
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    for record in records:
        for relation in record.causal_relations:
            source_key = relation.subject.casefold()
            target_key = relation.object.casefold()
            nodes.setdefault(
                source_key,
                {"id": source_key, "label": relation.subject, "dataset_id": dataset_id},
            )
            nodes.setdefault(
                target_key,
                {"id": target_key, "label": relation.object, "dataset_id": dataset_id},
            )
            edges.append(
                {
                    "id": relation.source_id
                    or f"{record.scenario_id}:{len(edges)}",
                    "source": source_key,
                    "target": target_key,
                    "label": relation.predicate,
                    "predicate": relation.predicate,
                    "scenario_id": record.scenario_id,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "is_expert_verified": relation.is_expert_verified,
                }
            )
            if len(edges) >= limit:
                break
        if len(edges) >= limit:
            break
    return {
        "dataset_id": dataset_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "truncated": len(records) >= limit,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
