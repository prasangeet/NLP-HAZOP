"""Hybrid reasoning: graph expansion, evidence adjudication, and model hypotheses."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import EvidenceItem
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import validate_relation_types
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
)

logger = logging.getLogger(__name__)


_ASSOCIATION_ORIGINS = frozenset(
    {
        RelationshipOrigin.SOURCE,
        RelationshipOrigin.RECONSTRUCTED,
        RelationshipOrigin.SYNTHETIC,
        RelationshipOrigin.EXPERT,
        RelationshipOrigin.RETRIEVAL_INFERRED,
        RelationshipOrigin.GRAPH_INFERRED,
        RelationshipOrigin.LLM_EXTRACTED,
    }
)

_EVIDENCE_RESULT_ORIGINS = _ASSOCIATION_ORIGINS - {
    RelationshipOrigin.SYNTHETIC,
}


class HybridReasoner:
    """Combine evidence-backed reasoning with independent model hypotheses."""

    def __init__(
        self,
        graph_reasoner: ReasoningStrategy,
        llm_reasoner: ReasoningStrategy,
        fallback_reasoner: ReasoningStrategy | None = None,
    ) -> None:
        self._graph_reasoner = graph_reasoner
        self._llm_reasoner = llm_reasoner
        self._fallback_reasoner = fallback_reasoner

    def reason_all(self, context: Any) -> tuple[CausalRelation, ...]:
        """Combine adjudicated associations and independent hypotheses."""

        if not self._is_supported_request(context):
            return ()

        dataset_id = getattr(context, "dataset_id", "default")
        known_endpoint = context.known_endpoint.strip().casefold()

        # 1. Collect graph-derived candidates.
        candidates = list(context.evidence_items)
        candidates.extend(self._collect_graph_candidates(context, dataset_id))

        # 2. Keep only candidates compatible with this request.
        compatible = tuple(
            item
            for item in candidates
            if self._is_eligible_candidate(
                item,
                context=context,
                dataset_id=dataset_id,
                known_endpoint=known_endpoint,
            )
        )

        logger.debug(
            "Scenario %s: %d total candidates, %d compatible candidates",
            context.scenario_id,
            len(candidates),
            len(compatible),
        )

        # 3. Adjudicate evidence using the correct epistemic path.
        supported_relations: list[CausalRelation] = []
        adjudication_context = replace(
            context,
            evidence_items=compatible,
        )

        if compatible:
            try:
                if context.predicate.strip().casefold() == "co_listed_with_consequence":
                    reason_all = getattr(self._llm_reasoner, "reason_all", None)

                    if callable(reason_all):
                        adjudicated = cast(
                            Iterable[CausalRelation],
                            reason_all(adjudication_context),
                        )

                        supported_relations = [
                            relation
                            for relation in adjudicated
                            if self._is_valid_evidence_result(
                                relation,
                                context=context,
                                dataset_id=dataset_id,
                                known_endpoint=known_endpoint,
                            )
                        ]
                else:
                    causal_result = self._llm_reasoner.reason(
                        adjudication_context
                    )
                    relation = causal_result.proposed_relation

                    if (
                        causal_result.status == ReasoningStatus.PROPOSED
                        and relation is not None
                        and self._is_valid_evidence_result(
                            relation,
                            context=context,
                            dataset_id=dataset_id,
                            known_endpoint=known_endpoint,
                        )
                    ):
                        supported_relations.append(relation)

            except Exception:
                logger.exception(
                    "Evidence adjudication failed for scenario %s",
                    context.scenario_id,
                )

        logger.debug(
            "Scenario %s: evidence adjudication returned %d valid relations",
            context.scenario_id,
            len(supported_relations),
        )

        # 4. Independently generate a model hypothesis.
        generated_relation = self._generate_hypothesis(
            context=context,
            dataset_id=dataset_id,
            known_endpoint=known_endpoint,
        )

        # 5. Merge the independent hypothesis without duplicating endpoints.
        if generated_relation is not None:
            existing_endpoints = {
                self._missing_value(
                    relation,
                    context.missing_endpoint,
                ).casefold()
                for relation in supported_relations
            }

            generated_endpoint = self._missing_value(
                generated_relation,
                context.missing_endpoint,
            ).casefold()

            if generated_endpoint and generated_endpoint not in existing_endpoints:
                supported_relations.append(generated_relation)

                logger.info(
                    "Accepted independent model hypothesis for scenario %s: %s",
                    context.scenario_id,
                    generated_endpoint,
                )
            else:
                logger.info(
                    "Suppressed duplicate model hypothesis for scenario %s: %s",
                    context.scenario_id,
                    generated_endpoint,
                )

        return tuple(supported_relations)

    def _collect_graph_candidates(
        self,
        context: Any,
        dataset_id: str,
    ) -> list[EvidenceItem]:
        """Convert graph-derived relations into explicitly inferred evidence."""

        graph_reason_all = getattr(self._graph_reasoner, "reason_all", None)

        if not callable(graph_reason_all):
            return []

        try:
            relations = cast(
                Iterable[CausalRelation],
                graph_reason_all(context),
            )
        except Exception:
            logger.exception(
                "Graph reasoning failed for scenario %s",
                context.scenario_id,
            )
            return []

        candidates: list[EvidenceItem] = []

        for relation in relations:
            if not relation.evidence.strip():
                continue

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
                        "Graph-composed row-association hypothesis; "
                        "not a direct source-verified causal edge."
                    ),
                    scenario_id=context.scenario_id,
                    dataset_id=dataset_id,
                    is_expert_verified=relation.is_expert_verified,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                    relevance_score=None,
                    subject_type=relation.subject_type,
                    object_type=relation.object_type,
                    source_location=getattr(
                        relation,
                        "source_location",
                        None,
                    ),
                )
            )

        return candidates

    @staticmethod
    def _is_supported_request(context: Any) -> bool:
        """Validate the request shape without hard-coding one predicate.

        The hybrid evidence-adjudication workflow is relation-agnostic: the
        requested predicate and endpoint direction are supplied by the
        recovery context. Restricting this check to one association predicate
        caused valid causal requests (for example, ``leads_to_consequence``)
        to bypass candidate adjudication entirely.
        """

        return (
            isinstance(getattr(context, "predicate", None), str)
            and bool(context.predicate.strip())
            and getattr(context, "missing_endpoint", None) in {"subject", "object"}
            and isinstance(getattr(context, "known_endpoint", None), str)
            and bool(context.known_endpoint.strip())
        )

    @staticmethod
    def _is_eligible_candidate(
        item: EvidenceItem,
        *,
        context: Any,
        dataset_id: str,
        known_endpoint: str,
    ) -> bool:
        """Check candidate provenance, dataset, evidence, and endpoint."""

        if item.origin not in _ASSOCIATION_ORIGINS:
            return False

        if item.dataset_id != dataset_id:
            return False

        if item.predicate.strip().casefold() != context.predicate.strip().casefold():
            return False

        if not item.evidence.strip():
            return False

        if (
            item.origin == RelationshipOrigin.EXPERT
            and not item.is_expert_verified
        ):
            return False

        if context.missing_endpoint == "object":
            return item.subject.strip().casefold() == known_endpoint

        if context.missing_endpoint == "subject":
            return item.object.strip().casefold() == known_endpoint

        return False

    @staticmethod
    def _is_valid_evidence_result(
        relation: CausalRelation,
        *,
        context: Any,
        dataset_id: str,
        known_endpoint: str,
    ) -> bool:
        """Validate an adjudicated result against its evidence request."""

        if relation.origin not in _EVIDENCE_RESULT_ORIGINS:
            return False

        if relation.dataset_id != dataset_id:
            return False

        if relation.predicate.strip().casefold() != context.predicate.strip().casefold():
            return False

        if not relation.subject.strip() or not relation.object.strip():
            return False

        if not relation.evidence.strip():
            return False

        if (
            relation.origin == RelationshipOrigin.EXPERT
            and not relation.is_expert_verified
        ):
            return False

        if context.missing_endpoint == "object":
            if relation.subject.strip().casefold() != known_endpoint:
                return False
        elif context.missing_endpoint == "subject":
            if relation.object.strip().casefold() != known_endpoint:
                return False
        else:
            return False

        return (
            validate_relation_types(
                relation.predicate,
                relation.subject_type,
                relation.object_type,
            )
            is None
        )

    def _generate_hypothesis(
        self,
        *,
        context: Any,
        dataset_id: str,
        known_endpoint: str,
    ) -> CausalRelation | None:
        """Generate and validate a hypothesis independently of retrieved evidence."""

        if self._fallback_reasoner is None:
            logger.warning(
                "No fallback reasoner configured for scenario %s",
                context.scenario_id,
            )
            return None

        generation_context = replace(
            context,
            evidence_items=(),
        )

        try:
            generated = self._fallback_reasoner.reason(generation_context)
        except Exception:
            logger.exception(
                "Independent hypothesis generation failed for scenario %s",
                context.scenario_id,
            )
            return None

        if generated.status != ReasoningStatus.PROPOSED:
            logger.info(
                "Independent model abstained for scenario %s: status=%s",
                context.scenario_id,
                generated.status,
            )
            return None

        relation = generated.proposed_relation

        if relation is None:
            logger.warning(
                "Model returned PROPOSED without a relation for scenario %s",
                context.scenario_id,
            )
            return None

        if not self._is_valid_model_hypothesis(
            relation,
            context=context,
            dataset_id=dataset_id,
            known_endpoint=known_endpoint,
        ):
            logger.warning(
                (
                    "Rejected model hypothesis for scenario %s: "
                    "%s -[%s]-> %s; origin=%s; dataset=%s"
                ),
                context.scenario_id,
                relation.subject,
                relation.predicate,
                relation.object,
                relation.origin,
                relation.dataset_id,
            )
            return None

        return relation

    @staticmethod
    def _missing_value(
        relation: CausalRelation | EvidenceItem,
        missing_endpoint: str,
    ) -> str:
        """Extract the endpoint missing from the input context."""

        if missing_endpoint == "object":
            return relation.object.strip()

        if missing_endpoint == "subject":
            return relation.subject.strip()

        return ""

    @staticmethod
    def _is_valid_model_hypothesis(
        relation: CausalRelation,
        *,
        context: Any,
        dataset_id: str,
        known_endpoint: str,
    ) -> bool:
        """Validate model provenance, dataset, direction, and ontology."""

        if relation.origin != RelationshipOrigin.MODEL_INFERRED:
            return False

        if relation.dataset_id != dataset_id:
            return False

        if relation.predicate.strip().casefold() != context.predicate.strip().casefold():
            return False

        if not relation.subject.strip() or not relation.object.strip():
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

        return (
            validate_relation_types(
                relation.predicate,
                relation.subject_type,
                relation.object_type,
            )
            is None
        )


    def reason(self, context: Any) -> ReasoningResult:
        """Run graph reasoning first, then use Gemini when needed.

        Association recovery uses the hybrid evidence-adjudication workflow.
        Other supported causal predicates use the graph result first and
        delegate to the LLM reasoner with the original evidence if graph
        reasoning cannot produce a proposal.
        """

        dataset_id = getattr(context, "dataset_id", "default")
        known_endpoint = context.known_endpoint.strip().casefold()

        # Graph reasoning remains independent of retrieved evidence.
        graph_context = replace(
            context,
            evidence_items=(),
        )
        graph_result = self._graph_reasoner.reason(graph_context)

        # Preserve a successful graph proposal.
        if graph_result.status == ReasoningStatus.PROPOSED:
            if graph_result.proposed_relation is not None:
                return graph_result

        # For predicates outside the association workflow, delegate
        # to Gemini using the original retrieved evidence.
        if not self._is_supported_request(context):
            if self._llm_reasoner is None:
                return graph_result

            try:
                llm_result = self._llm_reasoner.reason(context)
                return llm_result
            except Exception:
                logger.exception(
                    "LLM reasoning failed for scenario %s",
                    context.scenario_id,
                )
                return graph_result

        # Association-specific hybrid workflow.
        candidates = list(context.evidence_items)

        compatible = tuple(
            item
            for item in candidates
            if self._is_eligible_candidate(
                item,
                context=context,
                dataset_id=dataset_id,
                known_endpoint=known_endpoint,
            )
        )

        if not compatible:
            if self._fallback_reasoner is None:
                return graph_result

            try:
                fallback_result = self._fallback_reasoner.reason(
                    replace(context, evidence_items=())
                )
            except Exception:
                logger.exception(
                    "Fallback reasoning failed for scenario %s",
                    context.scenario_id,
                )
                return graph_result

            if fallback_result.status != ReasoningStatus.PROPOSED:
                return fallback_result

            relation = fallback_result.proposed_relation

            if relation is not None and self._is_valid_model_hypothesis(
                relation,
                context=context,
                dataset_id=dataset_id,
                known_endpoint=known_endpoint,
            ):
                return fallback_result

            logger.warning(
                "Rejected invalid fallback proposal for scenario %s",
                context.scenario_id,
            )

            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                dataset_id=dataset_id,
                supporting_evidence=fallback_result.supporting_evidence,
                abstention_reason=AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        llm_result = self._llm_reasoner.reason(
            replace(
                context,
                evidence_items=compatible,
            )
        )

        # Validate every LLM proposal. Consequence recovery is set-valued,
        # so validating only proposed_relation (the first item) silently
        # discarded all other supported consequences.
        if llm_result.status == ReasoningStatus.PROPOSED:
            valid_relations = tuple(
                relation
                for relation in llm_result.proposed_relations
                if self._is_valid_evidence_result(
                    relation,
                    context=context,
                    dataset_id=dataset_id,
                    known_endpoint=known_endpoint,
                )
            )
            if valid_relations:
                return replace(
                    llm_result,
                    proposed_relation=valid_relations[0],
                    proposed_relations=valid_relations,
                )

            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                dataset_id=dataset_id,
                supporting_evidence=llm_result.supporting_evidence,
                abstention_reason=AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        # If the LLM abstained (ambiguous or insufficient) and a fallback
        # reasoner is configured, try generating an independent hypothesis
        # rather than surfacing the abstention directly.
        if llm_result.status == ReasoningStatus.ABSTAINED and self._fallback_reasoner is not None:
            logger.info(
                "LLM abstained for scenario %s (reason=%s); trying fallback reasoner",
                context.scenario_id,
                llm_result.abstention_reason,
            )

            try:
                fallback_result = self._fallback_reasoner.reason(
                    replace(context, evidence_items=())
                )
            except Exception:
                logger.exception(
                    "Fallback reasoning failed for scenario %s",
                    context.scenario_id,
                )
                return llm_result

            if fallback_result.status != ReasoningStatus.PROPOSED:
                return llm_result

            relation = fallback_result.proposed_relation

            if relation is not None and self._is_valid_model_hypothesis(
                relation,
                context=context,
                dataset_id=dataset_id,
                known_endpoint=known_endpoint,
            ):
                return fallback_result

            logger.warning(
                "Rejected invalid fallback proposal for scenario %s after LLM abstention",
                context.scenario_id,
            )

        return llm_result
