"""Missing-link recovery endpoint over persisted HAZOP data."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_graph_repository
from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.retrieval.combined import CombinedRetriever
from app.application.strategies.retrieval.cross_scenario import CrossScenarioRetriever
from app.application.strategies.retrieval.graph_neighborhood import GraphNeighborhoodRetriever
from app.application.strategies.retrieval.scenario_local import ScenarioLocalRetriever
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.recover_missing_link import (
    MissingEndpoint,
    RecoverMissingLink,
    RecoveryRequest,
    RecoveryStatus,
)
from app.domain.repositories.graph_repository import DatasetRepository

router = APIRouter(tags=["recovery"])


class MissingLinkRequest(BaseModel):
    """Input query for a missing endpoint."""

    scenario_id: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    known_endpoint: str = Field(min_length=1, max_length=2000)
    missing_endpoint: MissingEndpoint
    limit: int = Field(default=20, ge=1, le=100)


@router.post(
    "/datasets/{dataset_id}/recovery",
    summary="Retrieve evidence and reason about a missing relationship endpoint",
)
def recover_missing_link(
    dataset_id: str,
    payload: MissingLinkRequest,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
) -> dict[str, object]:
    """Return retrieved candidates and a conservative graph reasoning result."""
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    local = ScenarioLocalRetriever(repository)
    cross = CrossScenarioRetriever(repository)
    combined = CombinedRetriever(local, cross)
    neighborhood = GraphNeighborhoodRetriever(combined, repository)
    recovery_use_case = RecoverMissingLink(neighborhood)
    request = RecoveryRequest(
        dataset_id=dataset_id,
        scenario_id=payload.scenario_id,
        predicate=payload.predicate,
        known_endpoint=payload.known_endpoint,
        missing_endpoint=payload.missing_endpoint,
        limit=payload.limit,
    )
    recovery = recovery_use_case.execute(request)
    if recovery.status == RecoveryStatus.SCENARIO_NOT_FOUND:
        raise HTTPException(status_code=404, detail="Scenario not found in this dataset.")

    evidence_context = BuildEvidenceContext().execute(recovery, request)
    reasoning = GraphReasoner().reason(evidence_context)
    return {
        "dataset_id": dataset_id,
        "scenario_id": payload.scenario_id,
        "recovery": recovery.to_dict(),
        "reasoning": reasoning.to_dict(),
        "notice": (
            "Recovered links are research hypotheses derived from worksheet "
            "associations and graph paths. They are not expert-validated causal facts."
        ),
    }
