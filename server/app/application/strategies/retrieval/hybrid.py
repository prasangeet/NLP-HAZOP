
"""Hybrid local-graph and TF-IDF retrieval strategy."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievalStrategy,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import RelationshipOrigin


class HybridRetriever:
    """Combine scenario-local evidence and vector-ranked row candidates.

    Scenario-local evidence receives priority. Cross-scenario candidates
    remain explicitly labelled and are never relabelled as local evidence.
    """

    def __init__(
        self,
        local_retriever: RetrievalStrategy,
        vector_retriever: RetrievalStrategy,
    ) -> None:
        self._local_retriever = local_retriever
        self._vector_retriever = vector_retriever

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        local = self._local_retriever.retrieve(request)

        if not local.scenario_exists:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
                graph_context=(),
            )

        vector = self._vector_retriever.retrieve(request)

        if (
            local.dataset_id != request.dataset_id
            or vector.dataset_id != request.dataset_id
        ):
            raise ValueError(
                "Hybrid retrievers must return the requested dataset_id."
            )

        if not vector.scenario_exists:
            raise ValueError(
                "Vector retriever reported a missing scenario that exists "
                "in the local retriever."
            )

        evidence: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        # Add local evidence first, followed by vector candidates.
        for item in (*local.evidence, *vector.evidence):
            if item.dataset_id != request.dataset_id:
                raise ValueError(
                    "Retrieved evidence must belong to the requested dataset."
                )

            key = (
                item.scenario_id.casefold(),
                item.relation.subject.casefold(),
                item.relation.predicate.casefold(),
                item.relation.object.casefold(),
            )

            if key in seen:
                continue

            seen.add(key)
            evidence.append(item)

        def priority(item: RetrievedEvidence) -> int:
            """Rank direct local evidence before approximate candidates."""
            if item.scope == RetrievalScope.SCENARIO_LOCAL:
                if (
                    item.relation.origin
                    != RelationshipOrigin.RETRIEVAL_INFERRED
                ):
                    return 0
                return 1

            return 2

        evidence.sort(
            key=lambda item: (
                priority(item),
                -(item.relevance_score or 0.0),
                item.scenario_id.casefold(),
            )
        )

        evidence = evidence[: request.limit]

        actual_scopes = {item.scope for item in evidence}

        if len(actual_scopes) > 1:
            scope = RetrievalScope.HYBRID
        elif actual_scopes:
            scope = next(iter(actual_scopes))
        else:
            scope = RetrievalScope.SCENARIO_LOCAL

        return RetrievalResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            scope=scope,
            evidence=tuple(evidence),
            scenario_exists=True,
            graph_context=local.graph_context,
            scenario_context=local.scenario_context,
            scenario_entities=local.scenario_entities,
        )
