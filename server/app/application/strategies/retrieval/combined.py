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
    """Prefer local evidence and fall back to other scenarios when needed.

    Cross-scenario retrieval is used only when local retrieval returns fewer
    results than the requested limit. Evidence retains its original scope
    and scenario provenance.
    """

    def __init__(
        self,
        local_retriever: RetrievalStrategy,
        cross_scenario_retriever: RetrievalStrategy,
    ) -> None:
        self._local_retriever = local_retriever
        self._cross_scenario_retriever = cross_scenario_retriever

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve local evidence first, then supplement it if necessary."""

        local_result = self._local_retriever.retrieve(request)

        # Do not search other scenarios if the target scenario doesn't exist.
        if not local_result.scenario_exists:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
            )

        local_evidence = list(local_result.evidence)

        # Local retrieval has already satisfied the requested limit.
        if len(local_evidence) >= request.limit:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=tuple(local_evidence[: request.limit]),
                scenario_exists=True,
            )

        remaining = request.limit - len(local_evidence)

        cross_result = self._cross_scenario_retriever.retrieve(request)

        # Deduplicate identical relationships while retaining distinct
        # relationships from different source scenarios.
        combined_evidence: list[RetrievedEvidence] = []
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
            combined_evidence.append(item)

            if len(combined_evidence) >= request.limit:
                break

        # Local evidence is prioritized, so cross-scenario evidence fills
        # only the remaining capacity.
        return RetrievalResult(
            scenario_id=request.scenario_id,
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=tuple(combined_evidence),
            scenario_exists=True,
        )
