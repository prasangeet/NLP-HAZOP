"""Tests for the deterministic reasoning baseline."""

import pytest

from app.application.strategies.deterministic_reasoner import (
    DeterministicReasoner,
)
from app.application.use_cases.build_evidence_context import (
    BuildEvidenceContext,
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import (
    RecoveryCandidate,
    RecoveryRequest,
    RecoveryResult,
    RecoveryStatus,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStatus,
)


def make_item(
    *,
    subject: str = "Inlet valve failure",
    predicate: str = "causes",
    object_: str = "High tank level",
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "Source identifies the valve failure.",
    confidence: float | None = 0.9,
    source_id: str | None = "source-001",
) -> EvidenceItem:
    """Create a test evidence item."""
    return EvidenceItem(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=evidence,
        confidence=confidence,
        source_id=source_id,
        relevance_note="Test evidence.",
    )


def make_context(*items: EvidenceItem) -> EvidenceContext:
    """Create an evidence context with the required query information."""
    return EvidenceContext(
        scenario_id="HAZOP-001",
        predicate="causes",
        known_endpoint="Inlet valve failure",
        missing_endpoint="object",
        status=(
            RecoveryStatus.CANDIDATES_FOUND
            if items
            else RecoveryStatus.NO_CANDIDATES
        ),
        evidence_items=items,
    )


def make_request(
    *,
    scenario_id: str = "HAZOP-001",
    predicate: str = "causes",
    known_endpoint: str = "Valve failure",
    missing_endpoint: str = "object",
) -> RecoveryRequest:
    """Create a missing-link recovery request."""
    return RecoveryRequest(
        scenario_id=scenario_id,
        predicate=predicate,
        known_endpoint=known_endpoint,
        missing_endpoint=missing_endpoint,
    )


def test_proposes_single_supported_relationship() -> None:
    context = make_context(make_item())

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Inlet valve failure"
    assert result.proposed_relation.object == "High tank level"
    assert result.proposed_relation.origin == RelationshipOrigin.SOURCE
    assert result.proposed_relation.source_id == "source-001"
    assert result.supporting_evidence == (
        "Source identifies the valve failure.",
    )


def test_abstains_when_context_is_empty() -> None:
    result = DeterministicReasoner().reason(make_context())

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE
    assert result.proposed_relation is None


def test_abstains_when_only_model_inferred_evidence_exists() -> None:
    context = make_context(
        make_item(origin=RelationshipOrigin.MODEL_INFERRED)
    )

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert (
        result.abstention_reason
        == AbstentionReason.INSUFFICIENT_SUPPORT
    )


def test_abstains_when_source_evidence_is_missing() -> None:
    context = make_context(make_item(evidence="   "))

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert (
        result.abstention_reason
        == AbstentionReason.INSUFFICIENT_SUPPORT
    )


def test_abstains_when_multiple_eligible_candidates_exist() -> None:
    context = make_context(
        make_item(object_="High tank level"),
        make_item(
            object_="Tank overflow",
            evidence="A second source supports this candidate.",
            source_id="source-002",
        ),
    )

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert (
        result.abstention_reason
        == AbstentionReason.AMBIGUOUS_EVIDENCE
    )
    assert len(result.supporting_evidence) == 2


def test_expert_evidence_can_support_a_proposal() -> None:
    context = make_context(
        make_item(origin=RelationshipOrigin.EXPERT)
    )

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.origin == RelationshipOrigin.EXPERT


def test_proposed_result_serializes_correctly() -> None:
    result = DeterministicReasoner().reason(
        make_context(make_item())
    )

    payload = result.to_dict()

    assert payload["status"] == "proposed"
    assert payload["scenario_id"] == "HAZOP-001"
    assert payload["abstention_reason"] is None
    assert payload["proposed_relation"]["source_id"] == "source-001"


def test_abstention_result_serializes_correctly() -> None:
    result = DeterministicReasoner().reason(make_context())

    payload = result.to_dict()

    assert payload["status"] == "abstained"
    assert payload["proposed_relation"] is None
    assert payload["abstention_reason"] == "no_evidence"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_reasoning_result_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError, match="confidence"):
        ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id="HAZOP-001",
            proposed_relation=CausalRelation(
                subject="A",
                predicate="causes",
                object="B",
                origin=RelationshipOrigin.SOURCE,
            ),
            confidence=confidence,
        )


def test_reasoning_result_rejects_proposal_without_relation() -> None:
    with pytest.raises(ValueError, match="relationship"):
        ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id="HAZOP-001",
        )


def test_reasoning_result_rejects_abstention_without_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id="HAZOP-001",
        )


def test_evidence_context_builder_integrates_with_reasoner() -> None:
    relation = CausalRelation(
        subject="Valve failure",
        predicate="causes",
        object="High level",
        origin=RelationshipOrigin.SOURCE,
        evidence="Documented in the source.",
        source_id="doc-17",
    )

    recovery = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(
            RecoveryCandidate(
                missing_value="High level",
                relation=relation,
            ),
        ),
    )

    request = make_request()
    context = BuildEvidenceContext().execute(recovery, request)
    result = DeterministicReasoner().reason(context)

    assert context.predicate == "causes"
    assert context.known_endpoint == "Valve failure"
    assert context.missing_endpoint == "object"
    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.source_id == "doc-17"


def test_abstains_when_candidate_predicate_does_not_match_query() -> None:
    context = make_context(make_item(predicate="increases"))

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_abstains_when_candidate_known_endpoint_does_not_match_query() -> None:
    context = make_context(make_item(subject="Different valve"))

    result = DeterministicReasoner().reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT
