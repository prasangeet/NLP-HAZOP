"""Missing-link recovery endpoint with graph, LLM and hybrid strategies."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import (
    get_ai_provider_factory,
    get_graph_repository,
)
from app.application.factories.application_factory import ApplicationFactory
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.recover_missing_link import (
    MissingEndpoint,
    RecoverMissingLink,
    RecoveryRequest,
    RecoveryStatus,
)
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.services.safety_validator import HazopSafetyValidator
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory

router = APIRouter(tags=["recovery"])


class ExcludedRelationRequest(BaseModel):
    """An exact relation to hide from the recovery reasoning pipeline."""

    scenario_id: str = Field(min_length=1, max_length=200)
    subject: str = Field(min_length=1, max_length=2000)
    predicate: str = Field(min_length=1, max_length=100)
    object: str = Field(min_length=1, max_length=2000)


class MissingLinkRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=100)
    known_endpoint: str = Field(min_length=1, max_length=2000)
    missing_endpoint: MissingEndpoint

    limit: int = Field(default=20, ge=1, le=100)
    strategy: Literal["graph", "llm", "hybrid"] = "hybrid"

    excluded_relations: list[ExcludedRelationRequest] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Exact relations to exclude from retrieval evidence and graph "
            "context during this recovery request."
        ),
    )


@router.post(
    "/datasets/{dataset_id}/recovery",
    summary="Recover a missing HAZOP link using graph, LLM or hybrid reasoning",
)
def recover_missing_link(
    dataset_id: str,
    payload: MissingLinkRequest,
    repository: Annotated[
        DatasetRepository,
        Depends(get_graph_repository),
    ],
    ai_factory: Annotated[
        AIProviderFactory | None,
        Depends(get_ai_provider_factory),
    ],
) -> dict[str, object]:
    """Retrieve evidence, reason about a missing link, and validate proposals."""

    if repository.get_dataset(dataset_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    excluded_relations = tuple(
        (
            relation.scenario_id,
            relation.subject,
            relation.predicate,
            relation.object,
        )
        for relation in payload.excluded_relations
    )

    request = RecoveryRequest(
        dataset_id=dataset_id,
        scenario_id=payload.scenario_id,
        predicate=payload.predicate,
        known_endpoint=payload.known_endpoint,
        missing_endpoint=payload.missing_endpoint,
        limit=payload.limit,
        excluded_relations=excluded_relations,
    )

    factory = ApplicationFactory(repository, ai_factory)

    try:
        retrieval_strategy = factory.create_retrieval_strategy()

        recovery = RecoverMissingLink(
            retrieval_strategy
        ).execute(request)

        if recovery.status == RecoveryStatus.SCENARIO_NOT_FOUND:
            raise HTTPException(
                status_code=404,
                detail="Scenario not found in this dataset.",
            )

        context = BuildEvidenceContext().execute(
            recovery,
            request,
        )

        reasoning_strategy = factory.create_reasoning_strategy(
            payload.strategy,
        )

        reasoning = reasoning_strategy.reason(context)

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    record = repository.get_record(
        payload.scenario_id,
        dataset_id=dataset_id,
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Scenario record not found in this dataset.",
        )

    proposed_relations = reasoning.proposed_relations

    if not proposed_relations:
        validation_result: dict[str, object] = {
            "status": "not_evaluated",
            "valid": None,
            "errors": [],
            "warnings": [
                (
                    "No reasoning proposal was produced; "
                    "safety validation was not performed."
                )
            ],
        }
    else:
        candidate_source_scenarios = {
            (
                candidate.subject.strip().casefold(),
                candidate.predicate.strip().casefold(),
                candidate.object.strip().casefold(),
            ): candidate.scenario_id.strip().casefold()
            for candidate in recovery.candidates
        }

        validations = [
            HazopSafetyValidator().validate_recovery_proposal(
                relation,
                predicate=payload.predicate,
                known_endpoint=payload.known_endpoint,
                missing_endpoint=payload.missing_endpoint.value,
                dataset_id=dataset_id,
                record=record,
                enforce_record_constraints=(
                    relation.origin.value != "graph_inferred"
                    and candidate_source_scenarios.get(
                        (
                            relation.subject.strip().casefold(),
                            relation.predicate.strip().casefold(),
                            relation.object.strip().casefold(),
                        )
                    )
                    in (None, payload.scenario_id.strip().casefold())
                ),
            )
            for relation in proposed_relations
        ]
        validation_result = {
            "status": (
                "validated"
                if all(validation.valid for validation in validations)
                else "rejected"
            ),
            "valid": all(validation.valid for validation in validations),
            "results": [validation.to_dict() for validation in validations],
            "errors": [
                error.to_dict()
                for validation in validations
                for error in validation.errors
            ],
            "warnings": [
                warning.to_dict()
                for validation in validations
                for warning in validation.warnings
            ],
        }

        if not validation_result["valid"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Reasoning proposal failed deterministic validation."
                    ),
                    "validation": validation_result,
                },
            )

    return {
        "dataset_id": dataset_id,
        "scenario_id": payload.scenario_id,
        "strategy": payload.strategy,
        "excluded_relations": [
            relation.model_dump()
            for relation in payload.excluded_relations
        ],
        "recovery": recovery.to_dict(),
        "reasoning": reasoning.to_dict(),
        "validation": validation_result,
        "notice": (
            "This is a research proposal, not an engineering approval. "
            "LLM-extracted, synthetic, retrieved and graph-derived evidence "
            "are not expert-validated causal facts or engineering approval."
        ),
    }
