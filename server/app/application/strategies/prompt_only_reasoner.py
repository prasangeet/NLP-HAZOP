"""LLM endpoint generation from scenario context, optionally graph-grounded."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from groq import APIError
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import (
    RELATION_SCHEMA,
    EntityType,
    infer_entity_types,
    validate_relation_types,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    EvidenceContextProtocol,
    ReasoningResult,
    ReasoningStatus,
)


class PromptOnlyChatModel(Protocol):
    """Minimal chat-model interface used by the prompt-only baseline."""

    def invoke(self, prompt: str) -> object:
        """Return a structured decision."""
        ...


class PromptOnlyDecision(BaseModel):
    """Structured prompt-only response."""

    decision: Literal["propose", "abstain"] = Field(
        description="Whether the scenario description supports a candidate."
    )
    missing_value: str | None = Field(
        default=None,
        description="The proposed missing endpoint, if supported.",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(description="Concise explanation.")


SERVER_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = SERVER_ROOT / ".env"
load_dotenv(ENV_FILE)


def _normalize_entity(value: str) -> str:
    """Normalize an entity phrase for conservative scenario-entity matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


class PromptOnlyReasoner:
    """Ask an LLM to complete a link from scenario context alone.

    No retrieved relationships or vector search are supplied. Typed scenario
    metadata is retained only as a guard against reusing a known entity of an
    incompatible type. The output is always marked MODEL_INFERRED.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        *,
        chat_model: PromptOnlyChatModel | None = None,
        include_graph_context: bool = False,
    ) -> None:
        self._include_graph_context = include_graph_context
        self._model_name = model_name
        self._temperature = temperature
        self._llm: PromptOnlyChatModel | None = chat_model

    def _get_chat_model(self) -> PromptOnlyChatModel:
        """Create the provider only when a generation request is actually made."""
        if self._llm is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"GROQ_API_KEY was not found. Add it to {ENV_FILE}."
                )
            self._llm = ChatGroq(
                model=self._model_name,
                temperature=self._temperature,
                api_key=SecretStr(api_key),
            ).with_structured_output(
                PromptOnlyDecision,
                method="json_schema",
                strict=False,
            )
        return self._llm

    @staticmethod
    def _expected_missing_types(
        predicate: str,
        missing_endpoint: str,
    ) -> frozenset[EntityType]:
        spec = RELATION_SCHEMA.get(predicate.strip().casefold())
        if spec is None:
            return frozenset()
        if missing_endpoint == "subject":
            return spec.subject_types
        if missing_endpoint == "object":
            return spec.object_types
        return frozenset()

    @staticmethod
    def _known_entity_type_error(
        candidate: str,
        scenario_entities: tuple[tuple[str, EntityType], ...],
        expected_types: frozenset[EntityType],
    ) -> str | None:
        """Reject a generated phrase matching a known entity of another type."""
        normalized_candidate = _normalize_entity(candidate)
        if len(normalized_candidate.split()) < 2:
            return None

        matches = [
            (value, entity_type)
            for value, entity_type in scenario_entities
            if normalized_candidate in _normalize_entity(value)
        ]
        if not matches:
            return None

        incompatible = [
            (value, entity_type)
            for value, entity_type in matches
            if entity_type not in expected_types
        ]
        if incompatible:
            details = ", ".join(
                f"{value!r} ({entity_type.value})"
                for value, entity_type in incompatible
            )
            expected = ", ".join(
                sorted(entity_type.value for entity_type in expected_types)
            )
            return (
                f"Generated endpoint {candidate!r} matches a known scenario "
                f"entity with incompatible type: {details}. Expected endpoint "
                f"type: {expected or 'not defined'}."
            )
        return None

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return a model-inferred candidate or abstain."""

        dataset_id = getattr(context, "dataset_id", "default")
        if not context.scenario_context.strip() and not (
            self._include_graph_context and context.graph_context
        ):
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
                dataset_id=dataset_id,
            )

        graph_context = "\n".join(
            f"- {item.subject} --{item.predicate}--> {item.object} "
            f"(origin={item.origin.value}; evidence={item.evidence})"
            for item in context.graph_context
        ) if self._include_graph_context else ""
        context_label = (
            "Scenario and graph context"
            if self._include_graph_context
            else "Scenario context only"
        )
        scenario_entities = tuple(getattr(context, "scenario_entities", ()))
        entity_catalog = "\n".join(
            f"- {value} [type={entity_type.value}]"
            for value, entity_type in scenario_entities
        ) or "No typed scenario-entity inventory supplied."
        expected_types = self._expected_missing_types(
            context.predicate,
            context.missing_endpoint,
        )
        expected_type_text = (
            ", ".join(sorted(item.value for item in expected_types))
            or "No ontology type is defined for this endpoint."
        )
        prompt = f"""
You are a cautious assistant for missing-link research in HAZOP records.

Task:
- Scenario ID: {context.scenario_id}
- Predicate: {context.predicate}
- Known endpoint: {context.known_endpoint}
- Missing endpoint: {context.missing_endpoint}
- Required missing-endpoint type: {expected_type_text}

{context_label}:
{context.scenario_context}

Typed entities already present in this scenario:
{entity_catalog}

Graph relationships (if supplied):
{graph_context or "None supplied."}

The missing endpoint MUST have the required ontology type. Do not return an
equipment item, process unit, parameter, or deviation when the required type
is Consequence. In particular, do not copy an entity from the typed scenario
inventory unless its listed type is one of the required missing-endpoint types.
Do not treat graph-derived paths or row co-occurrence as direct causal evidence.
Propose a missing endpoint only if it follows clearly from the scenario
description. Do not invent plant design details, safeguards, or causal facts.
A HAZOP row association is not proof of causation. If context is inadequate
or multiple answers are plausible, abstain. Return decision="propose" with
one missing_value, or decision="abstain" with missing_value=null.
""".strip()

        try:
            response = self._get_chat_model().invoke(prompt)
            decision = (
                response
                if isinstance(response, PromptOnlyDecision)
                else PromptOnlyDecision.model_validate(response)
            )
        except (
            APIError,
            OutputParserException,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (f"llm_call_failed:{type(exc).__name__}",),
                dataset_id=dataset_id,
            )

        if decision.decision != "propose":
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=dataset_id,
            )

        missing = (decision.missing_value or "").strip()
        if not missing:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=dataset_id,
            )

        if context.missing_endpoint == "object":
            subject, object_ = context.known_endpoint, missing
        elif context.missing_endpoint == "subject":
            subject, object_ = missing, context.known_endpoint
        else:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id=dataset_id,
            )

        if expected_types:
            type_error = self._known_entity_type_error(
                missing,
                scenario_entities,
                expected_types,
            )
            if type_error:
                return self._abstain(
                    context.scenario_id,
                    AbstentionReason.INSUFFICIENT_SUPPORT,
                    (type_error,),
                    dataset_id=dataset_id,
                )

        subject_type, object_type = infer_entity_types(
            context.predicate,
            subject,
            object_,
        )
        type_error = validate_relation_types(
            context.predicate,
            subject_type,
            object_type,
        )
        if type_error:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (type_error,),
                dataset_id=dataset_id,
            )

        relation = CausalRelation(
            subject=subject,
            predicate=context.predicate,
            object=object_,
            origin=RelationshipOrigin.MODEL_INFERRED,
            evidence=(
                "LLM-generated hypothesis based on "
                + (
                    "scenario and graph context. "
                    if self._include_graph_context
                    else "scenario context only. "
                )
                + decision.reason.strip()
            ),
            confidence=decision.confidence,
            dataset_id=dataset_id,
            subject_type=subject_type,
            object_type=object_type,
        )
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            dataset_id=dataset_id,
            proposed_relation=relation,
            supporting_evidence=(),
            confidence=decision.confidence,
        )

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
        supporting_evidence: tuple[str, ...] = (),
        dataset_id: str = "default",
    ) -> ReasoningResult:
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            supporting_evidence=supporting_evidence,
            abstention_reason=reason,
        )
