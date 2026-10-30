"""Enrich query retrieval with a bounded scenario-family graph context."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedEvidence,
    RetrievalScope,
    RetrievalStrategy,
)
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.repositories.graph_repository import GraphRepository


class GraphNeighborhoodRetriever:
    """Attach nearby graph edges from the same plant/process family.

    The target scenario is excluded from the cross-scenario neighborhood to
    prevent held-out target edges from leaking into the reasoning context.
    When plant_unit is available, the family is restricted to that unit;
    otherwise it falls back to the exact equipment label.
    """

    def __init__(
        self,
        base_retriever: RetrievalStrategy,
        repository: GraphRepository,
        *,
        max_records: int = 10_000,
    ) -> None:
        if isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be positive")
        self._base_retriever = base_retriever
        self._repository = repository
        self._max_records = max_records

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        result = self._base_retriever.retrieve(request)
        if not result.scenario_exists:
            return result

        target = self._repository.get_record(request.scenario_id)
        if target is None:
            return result

        records = self._repository.find_records(limit=self._max_records)
        if target.plant_unit.strip():
            family_records = [
                record
                for record in records
                if record.plant_unit.casefold() == target.plant_unit.casefold()
            ]
        else:
            family_records = [
                record
                for record in records
                if record.equipment.casefold() == target.equipment.casefold()
            ]

        edges: list[RetrievedEvidence] = list(result.graph_context)
        seen = {
            (
                item.scenario_id.casefold(),
                item.relation.subject.casefold(),
                item.relation.predicate.casefold(),
                item.relation.object.casefold(),
            )
            for item in edges
        }
        for record in family_records:
            if record.scenario_id.casefold() == request.scenario_id.casefold():
                continue
            for relation in record.causal_relations:
                if relation.origin in {
                    RelationshipOrigin.MODEL_INFERRED,
                    RelationshipOrigin.RETRIEVAL_INFERRED,
                    RelationshipOrigin.GRAPH_INFERRED,
                }:
                    continue
                key = (
                    record.scenario_id.casefold(),
                    relation.subject.casefold(),
                    relation.predicate.casefold(),
                    relation.object.casefold(),
                )
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    RetrievedEvidence(
                        scenario_id=record.scenario_id,
                        relation=relation,
                        scope=RetrievalScope.CROSS_SCENARIO,
                    )
                )

        return RetrievalResult(
            scenario_id=result.scenario_id,
            scope=result.scope,
            evidence=result.evidence,
            scenario_exists=True,
            graph_context=tuple(edges),
            scenario_context=result.scenario_context,
        )
