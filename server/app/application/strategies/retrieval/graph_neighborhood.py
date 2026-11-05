
"""Enrich query retrieval with a bounded scenario-family graph context."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievalStrategy,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import (
    RecordType,
    RelationshipOrigin,
)
from app.domain.repositories.graph_repository import GraphRepository


class GraphNeighborhoodRetriever:
    """Attach provenance-qualified edges from the same plant/process family.

    The target scenario is excluded from cross-scenario context to prevent
    held-out target edges from leaking into the reasoning context.

    Cross-scenario evidence is restricted to source, reconstructed, and
    expert-origin relationships. Synthetic, unspecified, and inferred
    relationships are excluded.
    """

    _ALLOWED_CROSS_SCENARIO_ORIGINS = frozenset(
        {
            RelationshipOrigin.SOURCE,
            RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.EXPERT,
        }
    )

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

        target = self._repository.get_record(
            request.scenario_id,
            dataset_id=request.dataset_id,
        )

        if target is None:
            return result

        records = self._repository.find_records(
            dataset_id=request.dataset_id,
            limit=self._max_records,
        )

        if target.plant_unit.strip():
            family_records = [
                record
                for record in records
                if record.plant_unit.strip().casefold()
                == target.plant_unit.casefold()
            ]
        else:
            family_records = [
                record
                for record in records
                if record.equipment.strip().casefold()
                == target.equipment.casefold()
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
            if record.dataset_id != request.dataset_id:
                continue

            if (
                record.scenario_id.casefold()
                == request.scenario_id.casefold()
            ):
                continue

            # Synthetic records must not enter cross-scenario evidence.
            if record.record_type == RecordType.SYNTHETIC:
                continue

            for relation in record.causal_relations:
                if (
                    relation.origin
                    not in self._ALLOWED_CROSS_SCENARIO_ORIGINS
                ):
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
                        dataset_id=record.dataset_id,
                        relation=relation,
                        scope=RetrievalScope.CROSS_SCENARIO,
                    )
                )

        return RetrievalResult(
            scenario_id=result.scenario_id,
            dataset_id=request.dataset_id,
            scope=result.scope,
            evidence=result.evidence,
            scenario_exists=True,
            graph_context=tuple(edges),
            scenario_context=result.scenario_context,
        )
