
"""Hybrid reasoning: graph expansion, evidence adjudication, and model hypotheses."""

from __future__ import annotations

import logging
from dataclasses import replace

from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import EvidenceItem
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import validate_relation_types
from app.domain.reasoning.models import (
    ReasoningResult,
    ReasoningStatus,
    ReasoningStrategy,
)

logger = logging.getLogger(__name__)


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

    def reason_all(self, context) -> tuple[CausalRelation, ...]:
        """Combine graph, retrieval, adjudicated, and model-generated relations.

        The method maintains two separate reasoning paths:

        1. Evidence-backed reasoning:
           Graph and retrieval candidates are passed to the LLM adjudicator.

        2. Independent hypothesis generation:
           The fallback model receives the scenario context without evidence
           candidates and may independently propose a missing endpoint.

        The two paths are combined after validation.
        """

        if context.predicate.strip().casefold() != "co_listed_with_consequence":
            return ()

        dataset_id = getattr(context, "dataset_id", "default")
        known_endpoint = context.known_endpoint.strip().casefold()

        # ---------------------------------------------------------
        # 1. Collect graph-derived candidates.
        # ---------------------------------------------------------

        graph_relations = ()
        graph_reason_all = getattr(
            self._graph_reasoner,
            "reason_all",
            None,
        )

        if callable(graph_reason_all):
            try:
                graph_relations = graph_reason_all(context)
            except Exception:
                logger.exception(
                    "Graph reasoning failed for scenario %s",
                    context.scenario_id,
                )

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

        # ---------------------------------------------------------
        # 2. Filter evidence candidates for the requested endpoint.
        # ---------------------------------------------------------

        compatible = tuple(
            item
            for item in candidates
            if (
                getattr(item, "dataset_id", "default") == dataset_id
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
            )
        )

        logger.debug(
            "Scenario %s: %d total candidates, %d compatible candidates",
            context.scenario_id,
            len(candidates),
            len(compatible),
        )

        # ---------------------------------------------------------
        # 3. Adjudicate evidence-backed candidates.
        # ---------------------------------------------------------

        supported_relations: list[CausalRelation] = []

        reason_all = getattr(
            self._llm_reasoner,
            "reason_all",
            None,
        )

        if compatible and callable(reason_all):
            adjudication_context = replace(
                context,
                evidence_items=compatible,
            )

            try:
                supported_relations = list(
                    reason_all(adjudication_context)
                )
            except Exception:
                logger.exception(
                    "Evidence adjudication failed for scenario %s",
                    context.scenario_id,
                )

        logger.debug(
            "Scenario %s: evidence adjudication returned %d relations",
            context.scenario_id,
            len(supported_relations),
        )

        # ---------------------------------------------------------
        # 4. Independently generate a model hypothesis.
        # ---------------------------------------------------------

        generated_relation = self._generate_hypothesis(
            context=context,
            dataset_id=dataset_id,
            known_endpoint=known_endpoint,
        )

        # ---------------------------------------------------------
        # 5. Merge the independent hypothesis.
        # ---------------------------------------------------------

        if generated_relation is not None:
            # Only suppress a hypothesis if the endpoint is already
            # represented by an adjudicated relation.
            #
            # Do NOT compare against all compatible evidence candidates:
            # those candidates may have been rejected by the adjudicator.
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

            if (
                generated_endpoint
                and generated_endpoint not in existing_endpoints
            ):
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

    def _generate_hypothesis(
        self,
        *,
        context,
        dataset_id: str,
        known_endpoint: str,
    ) -> CausalRelation | None:
        """Run and validate independent model hypothesis generation."""

        if self._fallback_reasoner is None:
            logger.warning(
                "No fallback reasoner configured for scenario %s",
                context.scenario_id,
            )
            return None

        # Independent generation must not receive retrieved or graph-derived
        # evidence candidates. It receives the scenario context only.
        generation_context = replace(
            context,
            evidence_items=(),
        )

        try:
            generated = self._fallback_reasoner.reason(
                generation_context
            )
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
        """Extract the endpoint that is missing from the input context."""

        if missing_endpoint == "object":
            return relation.object.strip()

        return relation.subject.strip()

    @staticmethod
    def _is_valid_model_hypothesis(
        relation: CausalRelation,
        *,
        context,
        dataset_id: str,
        known_endpoint: str,
    ) -> bool:
        """Validate provenance, dataset, direction, evidence, and ontology."""

        # The relation must explicitly originate from independent model
        # generation.
        if relation.origin != RelationshipOrigin.MODEL_INFERRED:
            return False

        # Do not allow relations from another dataset.
        if relation.dataset_id != dataset_id:
            return False

        # Only the benchmark's target predicate is accepted.
        if (
            relation.predicate.strip().casefold()
            != context.predicate.strip().casefold()
        ):
            return False

        # A proposed relation must contain supporting text.
        if not relation.evidence.strip():
            return False

        # Validate the known endpoint and relation direction.
        if context.missing_endpoint == "object":
            if relation.subject.strip().casefold() != known_endpoint:
                return False

        elif context.missing_endpoint == "subject":
            if relation.object.strip().casefold() != known_endpoint:
                return False

        else:
            return False

        # Validate endpoint types against the ontology.
        return (
            validate_relation_types(
                relation.predicate,
                relation.subject_type,
                relation.object_type,
            )
            is None
        )

    def reason(self, context) -> ReasoningResult:
        """Retain the existing single-result reasoning interface."""

        graph_context = replace(
            context,
            evidence_items=(),
        )

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
                        "Indirect graph path hypothesis; not direct "
                        "source-verified evidence."
                    ),
                    scenario_id=context.scenario_id,
                    dataset_id=getattr(
                        context,
                        "dataset_id",
                        "default",
                    ),
                    is_expert_verified=relation.is_expert_verified,
                    scope=RetrievalScope.SCENARIO_LOCAL,
                    relevance_score=None,
                    subject_type=relation.subject_type,
                    object_type=relation.object_type,
                )
            )

        dataset_id = getattr(
            context,
            "dataset_id",
            "default",
        )

        known_endpoint = context.known_endpoint.strip().casefold()

        compatible = [
            item
            for item in candidates
            if (
                getattr(item, "dataset_id", "default") == dataset_id
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
            )
        ]

        if not compatible:
            if self._fallback_reasoner is not None:
                return self._fallback_reasoner.reason(
                    replace(
                        context,
                        evidence_items=(),
                    )
                )

            return graph_result

        return self._llm_reasoner.reason(
            replace(
                context,
                evidence_items=tuple(compatible),
            )
        )
