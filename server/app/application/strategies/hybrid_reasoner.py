"""Hybrid reasoning: graph path expansion followed by LLM adjudication."""

from __future__ import annotations

from dataclasses import replace

from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import EvidenceItem
from app.domain.entities.hazop_record import CausalRelation, RelationshipOrigin
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
        fallback_reasoner: ReasoningStrategy | None = None,
    ) -> None:
        self._graph_reasoner = graph_reasoner
        self._llm_reasoner = llm_reasoner
        self._fallback_reasoner = fallback_reasoner

    def reason_all(self, context) -> tuple[CausalRelation, ...]:
        """Adjudicate multiple candidates for row-level association recovery.

        This set-valued path is separate from reason(), which remains the
        conservative single-link reasoning contract.
        """
        graph_relations = ()
        graph_reason_all = getattr(self._graph_reasoner, "reason_all", None)
        if callable(graph_reason_all):
            graph_relations = graph_reason_all(context)

        candidates = list(context.evidence_items)
        for relation in graph_relations:
            candidates.append(EvidenceItem(
                subject=relation.subject, predicate=relation.predicate,
                object=relation.object, origin=RelationshipOrigin.GRAPH_INFERRED,
                evidence=relation.evidence, confidence=relation.confidence,
                source_id=relation.source_id,
                relevance_note=(
                    "Graph-composed row-association hypothesis; not a "
                    "direct source-verified causal edge."
                ),
                scenario_id=context.scenario_id,
                dataset_id=getattr(context, "dataset_id", "default"),
                is_expert_verified=relation.is_expert_verified,
                scope=RetrievalScope.SCENARIO_LOCAL, relevance_score=None,
                subject_type=relation.subject_type,
                object_type=relation.object_type,
                source_location=getattr(relation, "source_location", None),
            ))

        dataset_id = getattr(context, "dataset_id", "default")
        known = context.known_endpoint.strip().casefold()
        compatible = tuple(
            item for item in candidates
            if getattr(item, "dataset_id", "default") == dataset_id
            and item.predicate.strip().casefold() == context.predicate.strip().casefold()
            and bool(item.evidence.strip())
            and (
                item.subject.strip().casefold() == known
                if context.missing_endpoint == "object"
                else item.object.strip().casefold() == known
            )
            and (item.origin != RelationshipOrigin.EXPERT
                 or getattr(item, "is_expert_verified", False))
        )
        if not compatible:
            return ()
        adjudication_context = replace(context, evidence_items=compatible)
        reason_all = getattr(self._llm_reasoner, "reason_all", None)
        if not callable(reason_all):
            return ()
        return reason_all(adjudication_context)

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

        dataset_id = getattr(context, "dataset_id", "default")
        known_endpoint = context.known_endpoint.strip().casefold()
        compatible_candidates = [
            item
            for item in candidates
            if getattr(item, "dataset_id", "default") == dataset_id
            and item.predicate.strip().casefold()
            == context.predicate.strip().casefold()
            and bool(item.evidence.strip())
            and (
                item.subject.strip().casefold() == known_endpoint
                if context.missing_endpoint == "object"
                else item.object.strip().casefold() == known_endpoint
            )
            and (
                item.origin != RelationshipOrigin.EXPERT
                or getattr(item, "is_expert_verified", False)
            )
        ]

        if not compatible_candidates:
            if self._fallback_reasoner is not None:
                # Retrieval may return rows that are irrelevant to this exact
                # query. Treat "no compatible evidence" like an empty result
                # and invoke generative recovery with scenario + graph context.
                return self._fallback_reasoner.reason(
                    replace(context, evidence_items=())
                )
            return graph_result

        adjudication_context = replace(
            context,
            evidence_items=tuple(compatible_candidates),
        )
        return self._llm_reasoner.reason(adjudication_context)
