"""LLM endpoint generation from scenario context, optionally graph-grounded."""

from __future__ import annotations

import os
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
from app.domain.ontology.schema import infer_entity_types, validate_relation_types
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
        description="The proposed missing endpoint, if supported."
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(description="Concise explanation.")


SERVER_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = SERVER_ROOT / ".env"
load_dotenv(ENV_FILE)


class PromptOnlyReasoner:
    """Ask an LLM to complete a link from scenario context alone.

    No retrieved relationships, vector search, or graph neighborhood is
    supplied. The output is always marked MODEL_INFERRED.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
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
            ).with_structured_output(PromptOnlyDecision)
        return self._llm

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
        context_label = "Scenario and graph context" if self._include_graph_context else "Scenario context only"
        prompt = f"""
You are a cautious assistant for missing-link research in HAZOP records.

Task:
- Scenario ID: {context.scenario_id}
- Predicate: {context.predicate}
- Known endpoint: {context.known_endpoint}
- Missing endpoint: {context.missing_endpoint}

{context_label}:
{context.scenario_context}

Graph relationships (if supplied):
{graph_context or "None supplied."}

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
        ):
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id=dataset_id,
            )

        if decision.decision != "propose":
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=getattr(context, "dataset_id", "default"),
            )

        missing = (decision.missing_value or "").strip()
        if not missing:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=getattr(context, "dataset_id", "default"),
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

        subject_type, object_type = infer_entity_types(context.predicate, subject, object_)
        type_error = validate_relation_types(context.predicate, subject_type, object_type)
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
                "Prompt-only model output based solely on scenario context. "
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
