"""Unit tests for the LLM missing-link reasoning strategy."""

from dataclasses import replace

from app.application.strategies.llm_reasoner import LLMDecision, LLMReasoner
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.ontology.schema import EntityType
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
    subject_type: EntityType = EntityType.CONCEPT,
    object_type: EntityType = EntityType.CONCEPT,
    process_context: str = "",
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
        subject_type=subject_type,
        object_type=object_type,
        process_context=process_context,
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
    fake_llm = FakeLLM(response)
    return LLMReasoner(chat_model=fake_llm), fake_llm


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


def test_prompt_includes_target_process_and_causal_graph_context() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[0],
            confidence=0.8,
            reason="The candidate has a supplied causal path.",
        )
    )
    candidate = make_item(
        predicate="leads_to_consequence",
        object_="Fire",
        subject="Other than composition",
        subject_type=EntityType.DEVIATION,
        object_type=EntityType.CONSEQUENCE,
        process_context="Equipment: T-301 | Design intent: Store liquid hexane",
    )
    graph_edges = (
        EvidenceItem(
            subject="Other than composition",
            predicate="leads_to",
            object="Flammable vapour release",
            origin=RelationshipOrigin.RECONSTRUCTED,
            evidence="The reconstructed row links the deviation to the effect.",
            confidence=None,
            source_id="T301-015:effect",
            relevance_note="Reconstructed graph edge.",
            scenario_id="T301-015",
            subject_type=EntityType.DEVIATION,
            object_type=EntityType.EFFECT,
        ),
        EvidenceItem(
            subject="Flammable vapour release",
            predicate="leads_to_consequence",
            object="Fire",
            origin=RelationshipOrigin.RECONSTRUCTED,
            evidence="The reconstructed row links the effect to fire.",
            confidence=None,
            source_id="T301-015:consequence",
            relevance_note="Reconstructed graph edge.",
            scenario_id="T301-015",
            subject_type=EntityType.EFFECT,
            object_type=EntityType.CONSEQUENCE,
        ),
    )
    context = replace(
        make_context(candidate),
        predicate="leads_to_consequence",
        known_endpoint="Other than composition",
        scenario_context=(
            "Plant/unit: Hexane storage | Equipment: T-301 | "
            "Design intent: Receive and store liquid hexane | "
            "Known causes: Closed outlet valve | Safeguards: High-level alarm"
        ),
        graph_context=graph_edges,
    )

    result = reasoner.reason(context)

    assert result.status == ReasoningStatus.PROPOSED
    assert fake_llm.prompt is not None
    assert "Equipment: T-301" in fake_llm.prompt
    assert "Closed outlet valve" in fake_llm.prompt
    assert "Source process context: Equipment: T-301" in fake_llm.prompt
    assert "Other than composition -[leads_to]-> Flammable vapour release" in fake_llm.prompt
    assert "Flammable vapour release -[leads_to_consequence]-> Fire" in fake_llm.prompt
    assert "valid typed causal path" in fake_llm.prompt


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


def test_reason_all_recovers_multiple_row_associations_with_provenance() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(
            decision="propose",
            evidence_indices=[0, 1],
            confidence=0.91,
            reason="Both endpoints are explicitly represented in the evidence.",
        )
    )
    first = make_item(
        predicate="co_listed_with_consequence",
        object_="Overfilling",
        source_id="row-1",
        subject_type=EntityType.CAUSE,
        object_type=EntityType.CONSEQUENCE,
    )
    second = make_item(
        predicate="co_listed_with_consequence",
        object_="Loss of containment",
        source_id="row-2",
        subject_type=EntityType.CAUSE,
        object_type=EntityType.CONSEQUENCE,
    )
    context = replace(
        make_context(first, second),
        predicate="co_listed_with_consequence",
    )

    relations = reasoner.reason_all(context)

    assert fake_llm.called
    assert fake_llm.prompt is not None
    assert "not determining physical causation" in fake_llm.prompt
    assert [relation.object for relation in relations] == [
        "Overfilling", "Loss of containment"
    ]
    assert [relation.source_id for relation in relations] == ["row-1", "row-2"]
    assert all(
        relation.origin == RelationshipOrigin.SOURCE for relation in relations
    )


def test_reason_all_is_restricted_to_row_association_predicate() -> None:
    reasoner, fake_llm = make_reasoner(
        LLMDecision(decision="propose", evidence_indices=[0], reason="Evidence."),
    )
    result = reasoner.reason_all(make_context(make_item()))
    assert result == ()
    assert not fake_llm.called


def test_reason_all_rejects_wrong_ontology_types() -> None:
    reasoner, _ = make_reasoner(
        LLMDecision(decision="propose", evidence_indices=[0], reason="Evidence."),
    )
    item = make_item(
        predicate="co_listed_with_consequence",
        subject_type=EntityType.CAUSE,
        object_type=EntityType.EFFECT,
    )
    context = replace(
        make_context(item),
        predicate="co_listed_with_consequence",
    )
    assert reasoner.reason_all(context) == ()


def test_reason_remains_single_candidate_and_abstains_on_multiple_indices() -> None:
    reasoner, _ = make_reasoner(
        LLMDecision(
            decision="propose", evidence_indices=[0, 1],
            reason="Two endpoints are present.",
        )
    )
    result = reasoner.reason(
        make_context(make_item(), make_item(object_="Tank overflow"))
    )
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.AMBIGUOUS_EVIDENCE
