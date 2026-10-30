"""Combine scenario-local and cross-scenario evidence retrieval."""

from __future__ import annotations

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedEvidence,
    RetrievalScope,
    RetrievalStrategy,
)


class CombinedRetriever:
    """Prefer local evidence and supplement it with cross-scenario evidence."""

    def __init__(
        self,
        local_retriever: RetrievalStrategy,
        cross_scenario_retriever: RetrievalStrategy,
    ) -> None:
        self._local_retriever = local_retriever
        self._cross_scenario_retriever = cross_scenario_retriever

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        local_result = self._local_retriever.retrieve(request)
        if not local_result.scenario_exists:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
                graph_context=(),
            )

        local_evidence = list(local_result.evidence)
        if len(local_evidence) >= request.limit:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=tuple(local_evidence[: request.limit]),
                graph_context=local_result.graph_context,
            )

        cross_result = self._cross_scenario_retriever.retrieve(request)
        combined: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        for item in (*local_evidence, *cross_result.evidence):
            key = (
                item.scenario_id.casefold(),
                item.relation.subject.strip().casefold(),
                item.relation.predicate.strip().casefold(),
                item.relation.object.strip().casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(item)
            if len(combined) >= request.limit:
                break

        graph_context: list[RetrievedEvidence] = []
        graph_seen: set[tuple[str, str, str, str]] = set()
        for item in (*local_result.graph_context, *cross_result.graph_context):
            key = (
                item.scenario_id.casefold(),
                item.relation.subject.strip().casefold(),
                item.relation.predicate.strip().casefold(),
                item.relation.object.strip().casefold(),
            )
            if key not in graph_seen:
                graph_seen.add(key)
                graph_context.append(item)

        return RetrievalResult(
            scenario_id=request.scenario_id,
            scope=(
                RetrievalScope.HYBRID
                if cross_result.evidence
                else RetrievalScope.SCENARIO_LOCAL
            ),
            evidence=tuple(combined),
            scenario_exists=True,
            graph_context=tuple(graph_context),
        )
