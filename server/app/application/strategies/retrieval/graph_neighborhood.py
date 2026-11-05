
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
        max_cross_scenario_edges: int = 100,
        max_cross_scenario_hops: int = 2,
    ) -> None:
        for name, value in (
            ("max_records", max_records),
            ("max_cross_scenario_edges", max_cross_scenario_edges),
            ("max_cross_scenario_hops", max_cross_scenario_hops),
        ):
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive")

        self._base_retriever = base_retriever
        self._repository = repository
        self._max_records = max_records
        self._max_cross_scenario_edges = max_cross_scenario_edges
        self._max_cross_scenario_hops = max_cross_scenario_hops

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

        candidate_edges: list[RetrievedEvidence] = []
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

                candidate_edges.append(
                    RetrievedEvidence(
                        scenario_id=record.scenario_id,
                        dataset_id=record.dataset_id,
                        relation=relation,
                        scope=RetrievalScope.CROSS_SCENARIO,
                        process_context=_record_process_context(record),
                    )
                )

        # Do not expose the full process-family graph. Retain only directed
        # paths that start at the requested known endpoint; unrelated family
        # relations (for example, a different deviation) are noise rather
        # than evidence for this recovery request.
        frontier = {request.known_endpoint.strip().casefold()}
        for _ in range(self._max_cross_scenario_hops):
            next_frontier: set[str] = set()

            for item in candidate_edges:
                relation = item.relation
                subject = relation.subject.strip().casefold()
                object_ = relation.object.strip().casefold()

                if request.missing_endpoint == "object":
                    connected = subject in frontier
                    next_node = object_
                else:
                    connected = object_ in frontier
                    next_node = subject

                if not connected:
                    continue

                key = (
                    item.scenario_id.casefold(),
                    subject,
                    relation.predicate.casefold(),
                    relation.object.casefold(),
                )

                if key in seen:
                    continue

                seen.add(key)
                edges.append(item)
                next_frontier.add(next_node)

                if len(edges) >= self._max_cross_scenario_edges:
                    break

            if len(edges) >= self._max_cross_scenario_edges or not next_frontier:
                break

            frontier = next_frontier

        return RetrievalResult(
            scenario_id=result.scenario_id,
            dataset_id=request.dataset_id,
            scope=result.scope,
            evidence=result.evidence,
            scenario_exists=True,
            graph_context=tuple(edges),
            scenario_context=result.scenario_context,
            scenario_entities=result.scenario_entities,
        )



def _record_process_context(record) -> str:
    """Describe a source scenario without exposing unrelated hidden labels."""
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
