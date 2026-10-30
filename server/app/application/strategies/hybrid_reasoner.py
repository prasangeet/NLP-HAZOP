"""Hybrid reasoning: graph path expansion followed by LLM adjudication."""

from __future__ import annotations

from dataclasses import replace

from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import EvidenceItem
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.reasoning.models import (
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
)


class HybridReasoner:
    """Use vector candidates and graph paths as inputs to LLM adjudication.

    The graph step sees only the explicit graph context, never the vector
    candidates as if they were graph facts. The LLM then sees both the ranked
    retrieval candidates and any graph-derived path, with provenance labels.
    """

    def __init__(
        self,
        graph_reasoner: ReasoningStrategy,
        llm_reasoner: ReasoningStrategy,
    ) -> None:
        self._graph_reasoner = graph_reasoner
        self._llm_reasoner = llm_reasoner

    def reason(self, context) -> ReasoningResult:
        graph_context = replace(context, evidence_items=())
        graph_result = self._graph_reasoner.reason(graph_context)

        candidates = list(context.evidence_items)
        if (
            graph_result.status == ReasoningStatus.PROPOSED
            and graph_result.proposed_relation is not None
        ):
            relation = graph_result.proposed_relation
            candidates.append(
                EvidenceItem(
                    subject=relation.subject,
                    predicate=relation.predicate,
                    object=relation.object,
                    origin=RelationshipOrigin.GRAPH_INFERRED,
                    evidence=relation.evidence,
                    confidence=None,
                    source_id=relation.source_id,
                    relevance_note=(
                        "Indirect graph path hypothesis. This is not a "
                        "direct source-verified causal edge."
                    ),
                    scenario_id=context.scenario_id,
                    dataset_id=getattr(context, "dataset_id", "default"),
                    is_expert_verified=relation.is_expert_verified,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                    relevance_score=None,
                    subject_type=relation.subject_type,
                    object_type=relation.object_type,
                )
            )

        if not candidates:
            return graph_result

        adjudication_context = replace(
            context,
            evidence_items=tuple(candidates),
        )
        return self._llm_reasoner.reason(adjudication_context)
