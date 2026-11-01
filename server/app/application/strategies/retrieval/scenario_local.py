"""Retrieve query evidence and the target scenario's local graph."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.ontology.schema import EntityType
from app.domain.repositories.graph_repository import GraphRepository


class ScenarioLocalRetriever:
    """Retrieve direct candidates plus eligible edges from the target scenario."""

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        record = self._repository.get_record(
            request.scenario_id,
            dataset_id=request.dataset_id,
        )
        if record is None:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
                graph_context=(),
            )

        graph_context = tuple(
            RetrievedEvidence(
                scenario_id=record.scenario_id,
                dataset_id=record.dataset_id,
                relation=relation,
                scope=RetrievalScope.SCENARIO_LOCAL,
            )
            for relation in record.causal_relations
            if relation.origin != RelationshipOrigin.MODEL_INFERRED
        )

        results: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in record.causal_relations:
            if relation.origin == RelationshipOrigin.MODEL_INFERRED:
                continue
            if relation.predicate.casefold() != request.predicate.casefold():
                continue
            if not self._matches_endpoints(
                relation=relation,
                known_endpoint=request.known_endpoint.casefold(),
                missing_endpoint=request.missing_endpoint,
            ):
                continue

            key = (
                relation.subject.casefold(),
                relation.predicate.casefold(),
                relation.object.casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(
                RetrievedEvidence(
                    scenario_id=record.scenario_id,
                    dataset_id=record.dataset_id,
                    relation=relation,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                )
            )
            if len(results) >= request.limit:
                break

        scenario_entities = tuple(
            (value, entity_type)
            for value, entity_type in (
                (record.plant_unit, EntityType.PROCESS_UNIT),
                (record.equipment, EntityType.EQUIPMENT),
                (record.parameter, EntityType.PARAMETER),
                (record.deviation, EntityType.DEVIATION),
            )
            if value
        )

        return RetrievalResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=tuple(results),
            scenario_exists=True,
            graph_context=graph_context,
            scenario_context=" | ".join(
                value for value in (
                    record.plant_unit,
                    record.equipment,
                    record.design_intent,
                    record.parameter,
                    record.guideword,
                    record.deviation,
                ) if value
            ),
            scenario_entities=scenario_entities,
        )

    @staticmethod
    def _matches_endpoints(
        *,
        relation,
        known_endpoint: str,
        missing_endpoint: str,
    ) -> bool:
        if missing_endpoint == "object":
            return relation.subject.casefold() == known_endpoint
        if missing_endpoint == "subject":
            return relation.object.casefold() == known_endpoint
        raise ValueError("missing_endpoint must be 'subject' or 'object'.")
