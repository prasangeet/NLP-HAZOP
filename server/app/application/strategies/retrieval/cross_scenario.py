"""Retrieve matching relationships from other HAZOP scenarios."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.repositories.graph_repository import GraphRepository


class CrossScenarioRetriever:
    """Retrieve relationships from scenarios other than the target."""

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        records = self._repository.find_records(
            dataset_id=request.dataset_id,
            limit=10_000,
        )

        results: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        for record in records:
            if (
                record.dataset_id != request.dataset_id
                or record.scenario_id == request.scenario_id
            ):
                continue

            for relation in record.causal_relations:
                if relation.origin == RelationshipOrigin.MODEL_INFERRED:
                    continue

                if relation.predicate.casefold() != request.predicate.casefold():
                    continue

                if not self._matches_endpoints(relation, request):
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
                results.append(
                    RetrievedEvidence(
                        scenario_id=record.scenario_id,
                        dataset_id=record.dataset_id,
                        relation=relation,
                        scope=RetrievalScope.CROSS_SCENARIO,
                    )
                )

                if len(results) >= request.limit:
                    return RetrievalResult(
                        scenario_id=request.scenario_id,
                        dataset_id=request.dataset_id,
                        scope=RetrievalScope.CROSS_SCENARIO,
                        evidence=tuple(results),
                    )

        return RetrievalResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=tuple(results),
        )

    @staticmethod
    def _matches_endpoints(
        relation,
        request: RetrievalRequest,
    ) -> bool:
        if request.missing_endpoint == "object":
            return (
                relation.subject.casefold()
                == request.known_endpoint.casefold()
            )

        if request.missing_endpoint == "subject":
            return (
                relation.object.casefold()
                == request.known_endpoint.casefold()
            )

        raise ValueError(
            "missing_endpoint must be 'subject' or 'object'."
        )
