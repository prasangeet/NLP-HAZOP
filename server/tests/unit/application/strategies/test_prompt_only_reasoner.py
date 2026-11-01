"""Tests for the no-retrieval prompt-only LLM baseline."""

from app.application.strategies.prompt_only_reasoner import (
    PromptOnlyDecision,
    PromptOnlyReasoner,
)
from app.application.use_cases.build_evidence_context import EvidenceContext
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


class FakePromptModel:
    def __init__(self, response: PromptOnlyDecision) -> None:
        self.response = response
        self.called = False
        self.prompt = ""

    def invoke(self, prompt: str) -> PromptOnlyDecision:
        self.called = True
        self.prompt = prompt
        return self.response


def make_context(scenario_context: str) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        graph_context=(),
        scenario_context=scenario_context,
    )


def test_returns_model_inferred_proposal_from_scenario_context_only() -> None:
    model = FakePromptModel(
        PromptOnlyDecision(
            decision="propose",
            missing_value="High tank level",
            confidence=0.62,
            reason="The scenario context suggests a high-level deviation.",
        )
    )
    result = PromptOnlyReasoner(chat_model=model).reason(
        make_context("Storage | T-301 | HighLevel | High level")
    )

    assert model.called
    assert "Scenario context only" in model.prompt
    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.object == "High tank level"
    assert result.proposed_relation.origin == RelationshipOrigin.MODEL_INFERRED
    assert result.supporting_evidence == ()


def test_abstains_without_scenario_context_without_calling_model() -> None:
    model = FakePromptModel(
        PromptOnlyDecision(decision="abstain", reason="No context.")
    )
    result = PromptOnlyReasoner(chat_model=model).reason(make_context(""))

    assert not model.called
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE


def test_respects_prompt_only_abstention() -> None:
    model = FakePromptModel(
        PromptOnlyDecision(
            decision="abstain",
            reason="Several outcomes plausible.",
        )
    )
    result = PromptOnlyReasoner(chat_model=model).reason(
        make_context("Storage | T-301 | High level")
    )

    assert model.called
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT
