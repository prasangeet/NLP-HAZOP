"""Tests for conservative graph-based missing-link reasoning."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


def make_item(
    *,
    subject: str = "Inlet valve fails open",
    predicate: str = "causes",
    object_: str = "High tank level",
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "The source documents this causal relationship.",
    confidence: float | None = 0.9,
    source_id: str | None = "paper-row-12",
) -> EvidenceItem:
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


def make_context(
    *items: EvidenceItem,
    predicate: str = "causes",
    known_endpoint: str = "Inlet valve fails open",
    missing_endpoint: str = "object",
) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="HAZOP-001",
        predicate=predicate,
        known_endpoint=known_endpoint,
        missing_endpoint=missing_endpoint,
        status=(
            RecoveryStatus.CANDIDATES_FOUND
            if items
            else RecoveryStatus.NO_CANDIDATES
        ),
        evidence_items=items,
    )


def test_proposes_unique_source_supported_missing_object() -> None:
    result = GraphReasoner().reason(make_context(make_item()))

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Inlet valve fails open"
    assert result.proposed_relation.predicate == "causes"
    assert result.proposed_relation.object == "High tank level"
    assert result.proposed_relation.origin == RelationshipOrigin.SOURCE
    assert result.proposed_relation.source_id == "paper-row-12"


def test_recovers_missing_subject() -> None:
    item = make_item(subject="Blocked outlet", object_="High tank level")
    result = GraphReasoner().reason(
        make_context(
            item,
            known_endpoint="High tank level",
            missing_endpoint="subject",
        )
    )

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Blocked outlet"
    assert result.proposed_relation.object == "High tank level"


def test_matches_predicate_and_known_endpoint_case_insensitively() -> None:
    result = GraphReasoner().reason(
        make_context(
            make_item(),
            predicate="CAUSES",
            known_endpoint="inlet valve fails open",
        )
    )
    assert result.status == ReasoningStatus.PROPOSED


def test_abstains_when_evidence_is_empty() -> None:
    result = GraphReasoner().reason(make_context())
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE


def test_abstains_when_no_evidence_matches_query() -> None:
    result = GraphReasoner().reason(
        make_context(make_item(predicate="increases"))
    )
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_excludes_model_inferred_evidence() -> None:
    result = GraphReasoner().reason(
        make_context(make_item(origin=RelationshipOrigin.MODEL_INFERRED))
    )
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_excludes_evidence_without_supporting_text() -> None:
    result = GraphReasoner().reason(make_context(make_item(evidence="   ")))
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_abstains_when_distinct_missing_endpoints_are_supported() -> None:
    result = GraphReasoner().reason(
        make_context(
            make_item(object_="High tank level"),
            make_item(
                object_="Tank overflow",
                evidence="A separate source supports this endpoint.",
                source_id="paper-row-13",
            ),
        )
    )
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.AMBIGUOUS_EVIDENCE
    assert len(result.supporting_evidence) == 2


def test_duplicate_evidence_for_one_endpoint_is_not_ambiguous() -> None:
    result = GraphReasoner().reason(
        make_context(
            make_item(source_id="paper-row-12"),
            make_item(
                evidence="An independent source supports the same endpoint.",
                source_id="paper-row-13",
                confidence=0.95,
            ),
        )
    )
    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.object == "High tank level"
    assert len(result.supporting_evidence) == 2
    assert result.confidence == 0.95


def test_does_not_propose_an_edge_with_wrong_known_endpoint() -> None:
    result = GraphReasoner().reason(
        make_context(
            make_item(subject="Different valve"),
            known_endpoint="Inlet valve fails open",
        )
    )
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT
