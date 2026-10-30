"""Retrieve relationships from the requested HAZOP scenario only."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedEvidence,
    RetrievalScope,
)
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.repositories.graph_repository import GraphRepository


class ScenarioLocalRetriever:
    """Retrieve matching, evidence-backed relationships from one scenario."""

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve relationships matching the predicate and known endpoint.

        Only relationships belonging to the requested scenario are considered.
        Model-inferred relationships are excluded from the evidence baseline.
        """

        record = self._repository.get_record(request.scenario_id)

        if record is None:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
            )

        results: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str]] = set()

        requested_predicate = request.predicate.casefold()
        known_endpoint = request.known_endpoint.casefold()

        for relation in record.causal_relations:
            # Do not treat model-generated relationships as established evidence.
            if relation.origin == RelationshipOrigin.MODEL_INFERRED:
                continue

            # Match the requested relationship type, ignoring case.
            if relation.predicate.casefold() != requested_predicate:
                continue

            # Match the known endpoint while leaving the requested endpoint open.
            if not self._matches_endpoints(
                relation=relation,
                known_endpoint=known_endpoint,
                missing_endpoint=request.missing_endpoint,
            ):
                continue

            # Avoid duplicate relationships while preserving their first occurrence.
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
                    relation=relation,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                )
            )

            if len(results) >= request.limit:
                break

        return RetrievalResult(
            scenario_id=request.scenario_id,
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=tuple(results),
            scenario_exists=True,
        )

    @staticmethod
    def _matches_endpoints(
        *,
        relation,
        known_endpoint: str,
        missing_endpoint: str,
    ) -> bool:
        """Match the known endpoint for the requested missing endpoint."""

        if missing_endpoint == "object":
            # The subject is known; retrieve possible objects.
            return relation.subject.casefold() == known_endpoint

        if missing_endpoint == "subject":
            # The object is known; retrieve possible subjects.
            return relation.object.casefold() == known_endpoint

        raise ValueError(
            "missing_endpoint must be 'subject' or 'object'."
        )
