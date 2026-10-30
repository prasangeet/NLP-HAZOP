"""Unit tests for the LLM missing-link reasoning strategy."""

from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.application.strategies.llm_reasoner import LLMDecision, LLMReasoner
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


class FakeLLM:
    """Small deterministic stand-in for the chat model."""

    def __init__(self, response: LLMDecision) -> None:
        self.response = response
        self.prompt: str | None = None
        self.called = False

    def invoke(self, prompt: str) -> LLMDecision:
        self.called = True
        self.prompt = prompt
        return self.response


def make_item(
    *,
    subject: str = "Inlet valve fails open",
    predicate: str = "causes",
    object_: str = "High tank level",
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "The source documents this relationship.",
    source_id: str = "paper-row-12",
) -> EvidenceItem:
    return EvidenceItem(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=evidence,
        confidence=0.9,
        source_id=source_id,
        relevance_note="Test evidence.",
    )


def make_context(*items: EvidenceItem) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="HAZOP-001",
        predicate="causes",
        known_endpoint="Inlet valve fails open",
        missing_endpoint="object",
        status=(
            RecoveryStatus.CANDIDATES_FOUND
            if items
            else RecoveryStatus.NO_CANDIDATES
        ),
        evidence_items=items,
    )


def make_reasoner(response: LLMDecision) -> tuple[LLMReasoner, FakeLLM]:
    reasoner = object.__new__(LLMReasoner)
    fake_llm = FakeLLM(response)
    reasoner._llm = fake_llm
    return reasoner, fake_llm


def test_proposes_selected_evidence_and_preserves_provenance() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[0],
            confidence=0.82,
            reason="The source explicitly documents the relationship.",
        )
    )

    result = reasoner.reason(make_context(make_item()))

    assert fake_llm.called
    assert fake_llm.prompt is not None
    assert "Predicate: causes" in fake_llm.prompt
    assert "Known endpoint: Inlet valve fails open" in fake_llm.prompt
    assert "Missing endpoint: object" in fake_llm.prompt
    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.source_id == "paper-row-12"
    assert result.proposed_relation.origin == RelationshipOrigin.SOURCE
    assert result.confidence == 0.82


def test_does_not_call_llm_when_no_evidence_exists() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="abstain_insufficient",
            reason="No evidence.",
        )
    )

    result = reasoner.reason(make_context())

    assert not fake_llm.called
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE


def test_does_not_send_model_inferred_evidence_to_llm() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="abstain_insufficient",
            reason="No eligible evidence.",
        )
    )

    result = reasoner.reason(
        make_context(make_item(origin=RelationshipOrigin.MODEL_INFERRED))
    )

    assert not fake_llm.called
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_rejects_evidence_that_does_not_match_query() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[0],
            reason="Should not be reached.",
        )
    )

    result = reasoner.reason(
        make_context(make_item(subject="A different valve"))
    )

    assert not fake_llm.called
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_abstains_when_model_selects_multiple_evidence_items() -> None:
    reasoner, _ = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[0, 1],
            reason="Two candidates are plausible.",
        )
    )

    result = reasoner.reason(
        make_context(
            make_item(),
            make_item(
                object_="Tank overflow",
                evidence="A second source supports another endpoint.",
                source_id="paper-row-13",
            ),
        )
    )

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.AMBIGUOUS_EVIDENCE


def test_abstains_when_model_returns_out_of_range_evidence_index() -> None:
    reasoner, _ = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[4],
            reason="Invalid evidence reference.",
        )
    )

    result = reasoner.reason(make_context(make_item()))

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_respects_explicit_model_abstention() -> None:
    reasoner, _ = make_reasoner(
        LLMDecision(
            decision="abstain_ambiguous",
            reason="The evidence does not distinguish the candidates.",
        )
    )

    result = reasoner.reason(make_context(make_item()))

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.AMBIGUOUS_EVIDENCE
