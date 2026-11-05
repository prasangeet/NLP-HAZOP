
"""LLM endpoint generation from scenario context, optionally graph-grounded.

Includes:
- Structured logging for the complete LLM lifecycle.
- Conservative prompt-size estimation and trimming.
- Configurable input and output token budgets.
- Gemini API error and rate-limit diagnostics.
- Response, abstention, and ontology-validation diagnostics.

Token estimation uses a conservative character-based heuristic. It is not an
exact tokenizer and should not be treated as provider-reported token usage.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
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


logger = logging.getLogger(__name__)

SERVER_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = SERVER_ROOT / ".env"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LIMIT = 32768
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_INPUT_BUDGET = 10000
DEFAULT_SAFETY_MARGIN = 512

# Conservative estimate: approximately one token per three characters.
# This is deliberately more conservative than the common 4-character estimate.
# The budget is sized conservatively for Gemini's context window.
CHARS_PER_ESTIMATED_TOKEN = 3
GEMINI_MODEL = "gemini-3.5-flash-lite"


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid integer environment variable %s=%r; using %d",
            name,
            raw_value,
            default,
        )
        return default

    if value <= 0:
        logger.warning(
            "Environment variable %s must be positive; using %d",
            name,
            default,
        )
        return default

    return value


def _estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens from character count.

    This does not replace the model's actual tokenizer.
    """
    return math.ceil(len(text) / CHARS_PER_ESTIMATED_TOKEN)


def _normalize_entity(value: str) -> str:
    """Normalize an entity phrase for conservative scenario-entity matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


# ---------------------------------------------------------------------------
# Model interfaces
# ---------------------------------------------------------------------------


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

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    reason: str = Field(
        description="Concise explanation."
    )


# ---------------------------------------------------------------------------
# Prompt-only reasoner
# ---------------------------------------------------------------------------


class PromptOnlyReasoner:
    """Ask an LLM to complete a link from scenario context alone.

    No retrieved relationships or vector-search candidates are supplied.

    Typed scenario metadata is retained only as a guard against reusing a
    known entity of an incompatible type.

    The output is always marked MODEL_INFERRED.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        *,
        chat_model: PromptOnlyChatModel | None = None,
        include_graph_context: bool = False,
    ) -> None:
        self._include_graph_context = include_graph_context
        self._model_name = GEMINI_MODEL
        self._temperature = temperature
        self._llm: PromptOnlyChatModel | None = chat_model

        # Token budget configuration.
        self._context_limit = _read_positive_int(
            "HAZOP_LLM_CONTEXT_LIMIT",
            DEFAULT_CONTEXT_LIMIT,
        )

        self._max_output_tokens = _read_positive_int(
            "HAZOP_LLM_MAX_OUTPUT_TOKENS",
            DEFAULT_OUTPUT_TOKENS,
        )

        self._safety_margin = _read_positive_int(
            "HAZOP_LLM_TOKEN_SAFETY_MARGIN",
            DEFAULT_SAFETY_MARGIN,
        )

        calculated_input_budget = (
            self._context_limit
            - self._max_output_tokens
            - self._safety_margin
        )

        configured_input_budget = _read_positive_int(
            "HAZOP_LLM_INPUT_TOKEN_BUDGET",
            DEFAULT_INPUT_BUDGET,
        )

        self._input_token_budget = min(
            configured_input_budget,
            calculated_input_budget,
        )

        if self._input_token_budget <= 0:
            raise ValueError(
                "Invalid LLM token configuration: input budget is not positive. "
                "Check HAZOP_LLM_CONTEXT_LIMIT, "
                "HAZOP_LLM_MAX_OUTPUT_TOKENS, and "
                "HAZOP_LLM_TOKEN_SAFETY_MARGIN."
            )

        logger.info(
            "[LLM] Reasoner initialized | model=%s temperature=%.2f "
            "context_limit=%d input_budget=%d output_budget=%d "
            "safety_margin=%d graph_context=%s injected_model=%s",
            self._model_name,
            self._temperature,
            self._context_limit,
            self._input_token_budget,
            self._max_output_tokens,
            self._safety_margin,
            self._include_graph_context,
            chat_model is not None,
        )

    # -----------------------------------------------------------------------
    # Model initialization
    # -----------------------------------------------------------------------

    def _get_chat_model(self) -> PromptOnlyChatModel:
        """Create the Gemini provider lazily."""

        if self._llm is not None:
            return self._llm

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            logger.error(
                "[LLM] GEMINI_API_KEY is missing. Expected environment file: %s",
                ENV_FILE,
            )
            raise RuntimeError(
                f"GEMINI_API_KEY was not found. Add it to {ENV_FILE}."
            )

        logger.info(
            "[LLM] Initializing Gemini model | model=%s max_output_tokens=%d",
            self._model_name,
            self._max_output_tokens,
        )

        self._llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            google_api_key=SecretStr(api_key),
        ).with_structured_output(
            PromptOnlyDecision,
            method="json_schema",
        )

        return self._llm

    # -----------------------------------------------------------------------
    # Ontology helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _expected_missing_types(
        predicate: str,
        missing_endpoint: str,
    ) -> frozenset[EntityType]:
        """Return the ontology types allowed for the missing endpoint."""

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

    # -----------------------------------------------------------------------
    # Prompt construction and token-budget enforcement
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        *,
        scenario_id: str,
        predicate: str,
        known_endpoint: str,
        missing_endpoint: str,
        expected_type_text: str,
        context_label: str,
        scenario_context: str,
        entity_catalog: str,
        graph_context: str,
    ) -> str:
        """Build the complete prompt from its individual sections."""

        return f"""
You are a cautious assistant for missing-link research in HAZOP records.

Task:
- Scenario ID: {scenario_id}
- Predicate: {predicate}
- Known endpoint: {known_endpoint}
- Missing endpoint: {missing_endpoint}
- Required missing-endpoint type: {expected_type_text}

{context_label}:
{scenario_context}

Typed entities already present in this scenario:
{entity_catalog}

Graph relationships (if supplied):
{graph_context or "None supplied."}

Instructions:
1. The missing endpoint MUST have the required ontology type.
2. Do not return an equipment item, process unit, parameter, or deviation
   when the required type is Consequence.
3. Do not copy an entity from the typed scenario inventory unless its
   listed type is one of the required missing-endpoint types.
4. Do not treat graph-derived paths or row co-occurrence as direct
   causal evidence.
5. Propose a missing endpoint only if it follows clearly from the
   scenario description.
6. Do not invent plant design details, safeguards, or causal facts.
7. A HAZOP row association is not proof of causation.
8. If context is inadequate or multiple answers are plausible, abstain.

Return decision="propose" with one missing_value, or
decision="abstain" with missing_value=null.
""".strip()

    def _prepare_prompt(
        self,
        *,
        scenario_id: str,
        predicate: str,
        known_endpoint: str,
        missing_endpoint: str,
        expected_type_text: str,
        context_label: str,
        scenario_context: str,
        entity_lines: list[str],
        graph_lines: list[str],
    ) -> tuple[str, dict[str, int | bool]]:
        """Fit the prompt within the configured estimated input budget.

        Trimming priority:
        1. Remove graph context from the end.
        2. Remove typed entity inventory entries from the end.
        3. Shorten scenario context as a last resort.

        Scenario metadata and task instructions are never silently removed.
        """

        original_context_chars = len(scenario_context)
        original_entity_count = len(entity_lines)
        original_graph_count = len(graph_lines)

        current_context = scenario_context
        current_entities = list(entity_lines)
        current_graph = list(graph_lines)

        def build() -> str:
            return self._build_prompt(
                scenario_id=scenario_id,
                predicate=predicate,
                known_endpoint=known_endpoint,
                missing_endpoint=missing_endpoint,
                expected_type_text=expected_type_text,
                context_label=context_label,
                scenario_context=current_context,
                entity_catalog="\n".join(current_entities)
                or "Entity inventory omitted to fit the token budget.",
                graph_context="\n".join(current_graph),
            )

        prompt = build()

        # Remove graph context first: it is optional supporting information.
        while (
            _estimate_tokens(prompt) > self._input_token_budget
            and current_graph
        ):
            current_graph.pop()
            prompt = build()

        # Typed entities are useful for ontology checks, but secondary to
        # preserving the scenario description.
        while (
            _estimate_tokens(prompt) > self._input_token_budget
            and current_entities
        ):
            current_entities.pop()
            prompt = build()

        # Scenario context is shortened only after optional sections.
        while (
            _estimate_tokens(prompt) > self._input_token_budget
            and len(current_context) > 500
        ):
            new_length = max(500, int(len(current_context) * 0.80))

            if new_length >= len(current_context):
                break

            current_context = current_context[:new_length].rstrip()
            prompt = build()

        estimated_tokens = _estimate_tokens(prompt)

        metadata: dict[str, int | bool] = {
            "estimated_tokens": estimated_tokens,
            "prompt_chars": len(prompt),
            "original_context_chars": original_context_chars,
            "final_context_chars": len(current_context),
            "original_entity_count": original_entity_count,
            "final_entity_count": len(current_entities),
            "original_graph_count": original_graph_count,
            "final_graph_count": len(current_graph),
            "was_trimmed": (
                len(current_context) != original_context_chars
                or len(current_entities) != original_entity_count
                or len(current_graph) != original_graph_count
            ),
        }

        if estimated_tokens > self._input_token_budget:
            logger.error(
                "[LLM] Prompt exceeds configured budget even after trimming | "
                "scenario=%s estimated_tokens=%d budget=%d "
                "prompt_chars=%d",
                scenario_id,
                estimated_tokens,
                self._input_token_budget,
                len(prompt),
            )

            raise ValueError(
                "Prompt exceeds configured input token budget after trimming."
            )

        logger.info(
            "[LLM] Prompt prepared | scenario=%s estimated_tokens=%d/%d "
            "prompt_chars=%d trimmed=%s",
            scenario_id,
            estimated_tokens,
            self._input_token_budget,
            len(prompt),
            metadata["was_trimmed"],
        )

        logger.debug(
            "[LLM] Prompt trimming details | scenario=%s "
            "context_chars=%d->%d entities=%d->%d graph_items=%d->%d",
            scenario_id,
            original_context_chars,
            len(current_context),
            original_entity_count,
            len(current_entities),
            original_graph_count,
            len(current_graph),
        )

        return prompt, metadata

    # -----------------------------------------------------------------------
    # Main reasoning entry point
    # -----------------------------------------------------------------------

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return a model-inferred candidate or abstain."""

        started_at = time.perf_counter()

        scenario_id = context.scenario_id
        dataset_id = getattr(context, "dataset_id", "default")

        scenario_context = context.scenario_context.strip()
        scenario_entities = tuple(
            getattr(context, "scenario_entities", ())
        )

        graph_items = (
            tuple(context.graph_context)
            if self._include_graph_context
            else ()
        )

        logger.info(
            "[LLM] Starting hypothesis generation | scenario=%s "
            "dataset=%s model=%s predicate=%s known_endpoint=%s "
            "missing_endpoint=%s",
            scenario_id,
            dataset_id,
            self._model_name,
            context.predicate,
            context.known_endpoint,
            context.missing_endpoint,
        )

        logger.debug(
            "[LLM] Input summary | scenario=%s scenario_context_chars=%d "
            "typed_entities=%d graph_items=%d evidence_items=%d",
            scenario_id,
            len(scenario_context),
            len(scenario_entities),
            len(graph_items),
            len(context.evidence_items),
        )

        # Independent hypothesis generation must not receive retrieved
        # evidence candidates.
        if not scenario_context and not graph_items:
            logger.warning(
                "[LLM] Abstaining: no usable scenario context | scenario=%s",
                scenario_id,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.NO_EVIDENCE,
                dataset_id=dataset_id,
            )

        graph_lines = [
            (
                f"- {item.subject} --{item.predicate}--> {item.object} "
                f"(origin={item.origin.value}; evidence={item.evidence})"
            )
            for item in graph_items
        ]

        entity_lines = [
            f"- {value} [type={entity_type.value}]"
            for value, entity_type in scenario_entities
        ]

        expected_types = self._expected_missing_types(
            context.predicate,
            context.missing_endpoint,
        )

        expected_type_text = (
            ", ".join(sorted(item.value for item in expected_types))
            or "No ontology type is defined for this endpoint."
        )

        context_label = (
            "Scenario and graph context"
            if self._include_graph_context
            else "Scenario context only"
        )

        # Construct and enforce the prompt budget.
        try:
            prompt, prompt_metadata = self._prepare_prompt(
                scenario_id=scenario_id,
                predicate=context.predicate,
                known_endpoint=context.known_endpoint,
                missing_endpoint=context.missing_endpoint,
                expected_type_text=expected_type_text,
                context_label=context_label,
                scenario_context=scenario_context,
                entity_lines=entity_lines,
                graph_lines=graph_lines,
            )

        except ValueError as exc:
            logger.error(
                "[LLM] Prompt preparation failed | scenario=%s error=%s",
                scenario_id,
                exc,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                ("prompt_exceeds_token_budget",),
                dataset_id=dataset_id,
            )

        if prompt_metadata["was_trimmed"]:
            logger.warning(
                "[LLM] Prompt was trimmed to fit budget | scenario=%s "
                "estimated_tokens=%d budget=%d",
                scenario_id,
                prompt_metadata["estimated_tokens"],
                self._input_token_budget,
            )

        # -------------------------------------------------------------------
        # Invoke Gemini
        # -------------------------------------------------------------------

        request_started = time.perf_counter()

        try:
            chat_model = self._get_chat_model()

            logger.info(
                "[LLM] Sending Gemini request | scenario=%s model=%s "
                "estimated_input_tokens=%d max_output_tokens=%d",
                scenario_id,
                self._model_name,
                prompt_metadata["estimated_tokens"],
                self._max_output_tokens,
            )

            response = chat_model.invoke(prompt)

            request_latency = time.perf_counter() - request_started

            logger.info(
                "[LLM] Gemini request completed | scenario=%s latency=%.3fs",
                scenario_id,
                request_latency,
            )

            self._log_usage_metadata(
                response=response,
                scenario_id=scenario_id,
            )

            decision = (
                response
                if isinstance(response, PromptOnlyDecision)
                else PromptOnlyDecision.model_validate(response)
            )

        except GoogleAPIError as exc:
            request_latency = time.perf_counter() - request_started
            status_code = getattr(exc, "status_code", None)

            if status_code == 429:
                logger.error(
                    "[LLM] Gemini rate limit exceeded | scenario=%s "
                    "status=%s latency=%.3fs error=%s",
                    scenario_id,
                    status_code,
                    request_latency,
                    exc,
                )
            else:
                logger.exception(
                    "[LLM] Gemini API error | scenario=%s status=%s "
                    "latency=%.3fs",
                    scenario_id,
                    status_code,
                    request_latency,
                )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (f"gemini_api_error:{status_code or type(exc).__name__}",),
                dataset_id=dataset_id,
            )

        except (
            OutputParserException,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            request_latency = time.perf_counter() - request_started

            logger.exception(
                "[LLM] Response parsing or validation failed | "
                "scenario=%s exception=%s latency=%.3fs",
                scenario_id,
                type(exc).__name__,
                request_latency,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (f"llm_response_invalid:{type(exc).__name__}",),
                dataset_id=dataset_id,
            )

        except Exception as exc:
            request_latency = time.perf_counter() - request_started

            logger.exception(
                "[LLM] Unexpected model invocation failure | "
                "scenario=%s exception=%s latency=%.3fs",
                scenario_id,
                type(exc).__name__,
                request_latency,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (f"llm_call_failed:{type(exc).__name__}",),
                dataset_id=dataset_id,
            )

        # -------------------------------------------------------------------
        # Interpret model decision
        # -------------------------------------------------------------------

        logger.info(
            "[LLM] Model decision received | scenario=%s decision=%s "
            "confidence=%s reason=%s",
            scenario_id,
            decision.decision,
            decision.confidence,
            decision.reason,
        )

        if decision.decision != "propose":
            logger.warning(
                "[LLM] Model abstained | scenario=%s reason=%s",
                scenario_id,
                decision.reason,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=dataset_id,
            )

        missing = (decision.missing_value or "").strip()

        if not missing:
            logger.warning(
                "[LLM] Model returned propose without missing_value | "
                "scenario=%s",
                scenario_id,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (decision.reason,) if decision.reason.strip() else (),
                dataset_id=dataset_id,
            )

        # -------------------------------------------------------------------
        # Reconstruct relationship direction
        # -------------------------------------------------------------------

        if context.missing_endpoint == "object":
            subject = context.known_endpoint
            object_ = missing

        elif context.missing_endpoint == "subject":
            subject = missing
            object_ = context.known_endpoint

        else:
            logger.error(
                "[LLM] Invalid missing_endpoint value | scenario=%s value=%s",
                scenario_id,
                context.missing_endpoint,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                ("invalid_missing_endpoint",),
                dataset_id=dataset_id,
            )

        # -------------------------------------------------------------------
        # Validate against known scenario entities
        # -------------------------------------------------------------------

        if expected_types:
            type_error = self._known_entity_type_error(
                missing,
                scenario_entities,
                expected_types,
            )

            if type_error:
                logger.warning(
                    "[LLM] Hypothesis rejected: known entity type mismatch | "
                    "scenario=%s candidate=%s details=%s",
                    scenario_id,
                    missing,
                    type_error,
                )

                return self._abstain(
                    scenario_id,
                    AbstentionReason.INSUFFICIENT_SUPPORT,
                    (type_error,),
                    dataset_id=dataset_id,
                )

        # -------------------------------------------------------------------
        # Ontology validation
        # -------------------------------------------------------------------

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
            logger.warning(
                "[LLM] Hypothesis rejected: ontology validation failed | "
                "scenario=%s relation=%s -[%s]-> %s error=%s",
                scenario_id,
                subject,
                context.predicate,
                object_,
                type_error,
            )

            return self._abstain(
                scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                (type_error,),
                dataset_id=dataset_id,
            )

        # -------------------------------------------------------------------
        # Create MODEL_INFERRED relation
        # -------------------------------------------------------------------

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

        total_latency = time.perf_counter() - started_at

        logger.info(
            "[LLM] Accepted MODEL_INFERRED hypothesis | scenario=%s "
            "dataset=%s relation=%s -[%s]-> %s confidence=%s "
            "total_latency=%.3fs",
            scenario_id,
            dataset_id,
            subject,
            context.predicate,
            object_,
            decision.confidence,
            total_latency,
        )

        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            proposed_relation=relation,
            supporting_evidence=(),
            confidence=decision.confidence,
        )

    # -----------------------------------------------------------------------
    # Usage diagnostics
    # -----------------------------------------------------------------------

    @staticmethod
    def _log_usage_metadata(
        *,
        response: object,
        scenario_id: str,
    ) -> None:
        """Log provider-reported token usage when available."""

        usage = getattr(response, "usage_metadata", None)

        if usage is None:
            response_metadata = getattr(
                response,
                "response_metadata",
                {},
            )

            if isinstance(response_metadata, dict):
                usage = response_metadata.get("token_usage")

        if usage:
            logger.info(
                "[LLM] Provider token usage | scenario=%s usage=%s",
                scenario_id,
                usage,
            )
        else:
            logger.debug(
                "[LLM] Provider token usage unavailable | scenario=%s",
                scenario_id,
            )

    # -----------------------------------------------------------------------
    # Abstention helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
        supporting_evidence: tuple[str, ...] = (),
        dataset_id: str = "default",
    ) -> ReasoningResult:
        """Create a standardized abstention result."""

        logger.debug(
            "[LLM] Returning abstention | scenario=%s dataset=%s "
            "reason=%s details=%s",
            scenario_id,
            dataset_id,
            reason,
            supporting_evidence,
        )

        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            supporting_evidence=supporting_evidence,
            abstention_reason=reason,
        )
