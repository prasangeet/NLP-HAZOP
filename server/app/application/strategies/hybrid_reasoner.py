"""Hybrid reasoning: graph expansion, evidence adjudication, and model hypotheses."""

from __future__ import annotations

from dataclasses import replace

from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import EvidenceItem
from app.domain.entities.hazop_record import CausalRelation, RelationshipOrigin
from app.domain.ontology.schema import validate_relation_types
from app.domain.reasoning.models import (
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
)


class HybridReasoner:
    """Combine retrieval, graph, and independently generated hypotheses.

    Evidence-backed candidates retain their original provenance after LLM
    adjudication. A separate prompt-only reasoner may contribute one additional
    MODEL_INFERRED hypothesis; it is never relabelled as source evidence.
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
        """Recover evidence-backed row associations plus a distinct LLM hypothesis.

        The set-valued path is limited to row-level association recovery.
        Graph and retrieval candidates are adjudicated together. Only after
        that adjudication returns at least one supported candidate do we ask
        the independent prompt-only reasoner for a complementary hypothesis.
        """
        if context.predicate.strip().casefold() != "co_listed_with_consequence":
            return ()

        graph_relations = ()
        graph_reason_all = getattr(self._graph_reasoner, "reason_all", None)
        if callable(graph_reason_all):
            graph_relations = graph_reason_all(context)

        candidates = list(context.evidence_items)
        for relation in graph_relations:
            candidates.append(
                EvidenceItem(
                    subject=relation.subject,
                    predicate=relation.predicate,
                    object=relation.object,
                    origin=RelationshipOrigin.GRAPH_INFERRED,
                    evidence=relation.evidence,
                    confidence=relation.confidence,
                    source_id=relation.source_id,
                    relevance_note=(
                        "Graph-composed row-association hypothesis; not a "
                        "direct source-verified causal edge."
                    ),
                    scenario_id=context.scenario_id,
                    dataset_id=getattr(context, "dataset_id", "default"),
                    is_expert_verified=relation.is_expert_verified,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                    relevance_score=None,
                    subject_type=relation.subject_type,
                    object_type=relation.object_type,
                    source_location=getattr(relation, "source_location", None),
                )
            )

        dataset_id = getattr(context, "dataset_id", "default")
        known = context.known_endpoint.strip().casefold()
        compatible = tuple(
            item
            for item in candidates
            if getattr(item, "dataset_id", "default") == dataset_id
            and item.predicate.strip().casefold()
            == context.predicate.strip().casefold()
            and bool(item.evidence.strip())
            and (
                item.subject.strip().casefold() == known
                if context.missing_endpoint == "object"
                else item.object.strip().casefold() == known
            )
            and (
                item.origin != RelationshipOrigin.EXPERT
                or getattr(item, "is_expert_verified", False)
            )
        )
        if not compatible:
            return ()

        reason_all = getattr(self._llm_reasoner, "reason_all", None)
        if not callable(reason_all):
            return ()

        adjudication_context = replace(context, evidence_items=compatible)
        supported_relations = list(reason_all(adjudication_context))
        if not supported_relations or self._fallback_reasoner is None:
            return tuple(supported_relations)

        # Keep independent generation separate from evidence adjudication.
        # PromptOnlyReasoner consumes scenario/graph context, not these
        # retrieval candidates, and marks its result MODEL_INFERRED.
        generated = self._fallback_reasoner.reason(
            replace(context, evidence_items=())
        )
        if (
            generated.status != ReasoningStatus.PROPOSED
            or generated.proposed_relation is None
        ):
            return tuple(supported_relations)

        relation = generated.proposed_relation
        if not self._is_valid_model_hypothesis(
            relation,
            context=context,
            dataset_id=dataset_id,
            known_endpoint=known,
        ):
            return tuple(supported_relations)

        endpoint = (
            relation.object
            if context.missing_endpoint == "object"
            else relation.subject
        ).strip().casefold()
        # Do not duplicate a candidate that already exists in the evidence
        # pool, even if the adjudicator chose not to return it.
        existing_endpoints = {
            (
                item.object
                if context.missing_endpoint == "object"
                else item.subject
            ).strip().casefold()
            for item in compatible
        }
        existing_endpoints.update(
            (
                item.object
                if context.missing_endpoint == "object"
                else item.subject
            ).strip().casefold()
            for item in supported_relations
        )
        if endpoint and endpoint not in existing_endpoints:
            supported_relations.append(relation)

        return tuple(supported_relations)

    @staticmethod
    def _is_valid_model_hypothesis(
        relation: CausalRelation,
        *,
        context,
        dataset_id: str,
        known_endpoint: str,
    ) -> bool:
        """Validate the generated relation's direction, type, and provenance."""
        if relation.origin != RelationshipOrigin.MODEL_INFERRED:
            return False
        if relation.dataset_id != dataset_id:
            return False
        if relation.predicate.strip().casefold() != context.predicate.strip().casefold():
            return False
        if not relation.evidence.strip():
            return False
        if context.missing_endpoint == "object":
            if relation.subject.strip().casefold() != known_endpoint:
                return False
        elif context.missing_endpoint == "subject":
            if relation.object.strip().casefold() != known_endpoint:
                return False
        else:
            return False
        return validate_relation_types(
            relation.predicate,
            relation.subject_type,
            relation.object_type,
        ) is None

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
                return self._fallback_reasoner.reason(
                    replace(context, evidence_items=())
                )
            return graph_result

        adjudication_context = replace(
            context,
            evidence_items=tuple(compatible_candidates),
        )
        return self._llm_reasoner.reason(adjudication_context)
