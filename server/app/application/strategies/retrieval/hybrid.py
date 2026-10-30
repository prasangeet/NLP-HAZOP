"""Hybrid local-graph and TF-IDF retrieval strategy."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievalStrategy,
    RetrievedEvidence,
)


class HybridRetriever:
    """Combine scenario-local exact evidence and vector-ranked row candidates."""

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
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
                graph_context=(),
            )

        vector = self._vector_retriever.retrieve(request)
        evidence: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        # Keep direct local relationships ahead of approximate row retrieval.
        for item in (*local.evidence, *vector.evidence):
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
            if len(evidence) >= request.limit:
                break

        evidence.sort(
            key=lambda item: (
                0 if item.scope == RetrievalScope.SCENARIO_LOCAL
                and item.relation.origin.value != "retrieval_inferred" else 1,
                -(item.relevance_score or 0.0),
                item.scenario_id.casefold(),
            )
        )

        has_local = bool(local.evidence)
        has_vector = bool(vector.evidence)
        if has_local and has_vector:
            scope = RetrievalScope.HYBRID
        elif has_local:
            scope = RetrievalScope.SCENARIO_LOCAL
        elif has_vector:
            scope = vector.scope
        else:
            scope = RetrievalScope.SCENARIO_LOCAL

        return RetrievalResult(
            scenario_id=request.scenario_id,
            scope=scope,
            evidence=tuple(evidence[: request.limit]),
            scenario_exists=True,
            graph_context=local.graph_context,
            scenario_context=local.scenario_context,
        )
