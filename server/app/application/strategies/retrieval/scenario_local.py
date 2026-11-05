
"""Retrieve query evidence and the target scenario's local graph."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import EntityType
from app.domain.repositories.graph_repository import GraphRepository


class ScenarioLocalRetriever:
    """Retrieve eligible direct relationships from the target scenario."""

    _ALLOWED_LOCAL_ORIGINS = frozenset(
        {
            RelationshipOrigin.SOURCE,
            RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.EXPERT,
            RelationshipOrigin.SYNTHETIC,
        }
    )

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

        eligible_relations = tuple(
            relation
            for relation in record.causal_relations
            if relation.origin in self._ALLOWED_LOCAL_ORIGINS
        )

        process_context = _record_process_context(record)
        graph_context = tuple(
            RetrievedEvidence(
                scenario_id=record.scenario_id,
                dataset_id=record.dataset_id,
                relation=relation,
                scope=RetrievalScope.SCENARIO_LOCAL,
                process_context=process_context,
            )
            for relation in eligible_relations
        )

        results: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str]] = set()

        for relation in eligible_relations:
            if relation.predicate.casefold() != request.predicate.casefold():
                continue

            if not self._matches_endpoints(
                relation=relation,
                known_endpoint=request.known_endpoint,
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
                    process_context=process_context,
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

        scenario_context = process_context

        return RetrievalResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=tuple(results),
            scenario_exists=True,
            graph_context=graph_context,
            scenario_context=scenario_context,
            scenario_entities=scenario_entities,
        )

    @staticmethod
    def _matches_endpoints(
        *,
        relation: CausalRelation,
        known_endpoint: str,
        missing_endpoint: str,
    ) -> bool:
        known = known_endpoint.strip().casefold()

        if missing_endpoint == "object":
            return relation.subject.casefold() == known

        if missing_endpoint == "subject":
            return relation.object.casefold() == known

        raise ValueError(
            "missing_endpoint must be 'subject' or 'object'."
        )



def _record_process_context(record) -> str:
    """Build target context without including consequence labels that may be held out."""
    fields = (
        ("Plant/unit", record.plant_unit),
        ("Equipment", record.equipment),
        ("Design intent", record.design_intent),
        ("Parameter", record.parameter),
        ("Guideword", record.guideword),
        ("Deviation", record.deviation),
        ("Known causes", "; ".join(record.causes)),
        ("Known effects", "; ".join(record.effects)),
        ("Safeguards", "; ".join(record.safeguards)),
        ("Recommendations", "; ".join(record.recommendations)),
    )
    return " | ".join(f"{label}: {value}" for label, value in fields if value)
