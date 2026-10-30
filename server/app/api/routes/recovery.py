"""Missing-link recovery endpoint with graph, LLM and hybrid strategies."""
from __future__ import annotations
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from app.api.dependencies import get_graph_repository, get_ai_provider_factory
from app.application.factories.application_factory import ApplicationFactory
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.recover_missing_link import MissingEndpoint, RecoverMissingLink, RecoveryRequest, RecoveryStatus
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.services.safety_validator import HazopSafetyValidator
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory
router = APIRouter(tags=["recovery"])
class MissingLinkRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    known_endpoint: str = Field(min_length=1, max_length=2000)
    missing_endpoint: MissingEndpoint
    limit: int = Field(default=20, ge=1, le=100)
    strategy: Literal["graph", "llm", "hybrid"] = "hybrid"
@router.post("/datasets/{dataset_id}/recovery", summary="Recover a missing HAZOP link using graph, LLM or hybrid reasoning")
def recover_missing_link(
    dataset_id: str,
    payload: MissingLinkRequest,
    repository: Annotated[DatasetRepository, Depends(get_graph_repository)],
    ai_factory: Annotated[AIProviderFactory | None, Depends(get_ai_provider_factory)],
) -> dict[str, object]:
    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    request = RecoveryRequest(dataset_id=dataset_id, scenario_id=payload.scenario_id, predicate=payload.predicate,
        known_endpoint=payload.known_endpoint, missing_endpoint=payload.missing_endpoint, limit=payload.limit)
    factory = ApplicationFactory(repository, ai_factory)
    try:
        recovery = RecoverMissingLink(factory.create_retrieval_strategy()).execute(request)
        if recovery.status == RecoveryStatus.SCENARIO_NOT_FOUND:
            raise HTTPException(status_code=404, detail="Scenario not found in this dataset.")
        context = BuildEvidenceContext().execute(recovery, request)
        reasoning = factory.create_reasoning_strategy(payload.strategy).reason(context)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    validation = HazopSafetyValidator().validate_proposal(reasoning.proposed_relation) if reasoning.proposed_relation else None
    if validation is not None and not validation.valid:
        raise HTTPException(status_code=422, detail={"message":"Reasoning proposal failed deterministic validation.","validation":validation.to_dict()})
    return {
        "dataset_id": dataset_id, "scenario_id": payload.scenario_id, "strategy": payload.strategy,
        "recovery": recovery.to_dict(), "reasoning": reasoning.to_dict(),
        "validation": validation.to_dict() if validation else {"valid":True,"errors":[],"warnings":[]},
        "notice": "This is a research proposal, not an engineering approval. LLM-extracted, synthetic, retrieved and graph-derived evidence is not expert-validated causal truth.",
    }
