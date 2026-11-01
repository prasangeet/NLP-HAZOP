"""Tests for conservative graph-based missing-link reasoning."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.ontology.schema import EntityType
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



def _typed_edge(
    subject: str,
    predicate: str,
    object_: str,
    subject_type: EntityType,
    object_type: EntityType,
    source_id: str,
) -> EvidenceItem:
    return EvidenceItem(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=RelationshipOrigin.RECONSTRUCTED,
        evidence=f"Reconstructed row association: {subject} {predicate} {object_}.",
        confidence=None,
        source_id=source_id,
        relevance_note="Reconstructed graph evidence.",
        dataset_id="t301-reconstructed",
        subject_type=subject_type,
        object_type=object_type,
    )


def _cause_consequence_context(*edges: EvidenceItem) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="T301-TEST",
        predicate="co_listed_with_consequence",
        known_endpoint="Blocked inflow line",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        graph_context=edges,
        dataset_id="t301-reconstructed",
    )


def test_recovers_t301_cause_to_consequence_from_two_hop_path() -> None:
    edges = (
        _typed_edge(
            "Blocked inflow line",
            "causes_deviation",
            "Low level",
            EntityType.CAUSE,
            EntityType.DEVIATION,
            "row-1-cause",
        ),
        _typed_edge(
            "Low level",
            "leads_to_consequence",
            "Pump running dry",
            EntityType.DEVIATION,
            EntityType.CONSEQUENCE,
            "row-1-consequence",
        ),
    )

    result = GraphReasoner(max_hops=2).reason(
        _cause_consequence_context(*edges)
    )

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Blocked inflow line"
    assert result.proposed_relation.predicate == "co_listed_with_consequence"
    assert result.proposed_relation.object == "Pump running dry"
    assert result.proposed_relation.subject_type == EntityType.CAUSE
    assert result.proposed_relation.object_type == EntityType.CONSEQUENCE
    assert result.proposed_relation.origin == RelationshipOrigin.GRAPH_INFERRED
    assert "does not establish an independently validated causal assertion" in result.proposed_relation.evidence


def test_recovers_cause_to_consequence_through_intermediate_effect() -> None:
    edges = (
        _typed_edge(
            "Blocked inflow line",
            "causes_deviation",
            "Low level",
            EntityType.CAUSE,
            EntityType.DEVIATION,
            "row-1-cause",
        ),
        _typed_edge(
            "Low level",
            "leads_to",
            "Pump cavitation",
            EntityType.DEVIATION,
            EntityType.EFFECT,
            "row-1-effect",
        ),
        _typed_edge(
            "Pump cavitation",
            "leads_to_consequence",
            "Pump running dry",
            EntityType.EFFECT,
            EntityType.CONSEQUENCE,
            "row-1-consequence",
        ),
    )

    result = GraphReasoner(max_hops=3).reason(
        _cause_consequence_context(*edges)
    )

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.object == "Pump running dry"
    assert "causes_deviation" in result.proposed_relation.evidence
    assert "leads_to" in result.proposed_relation.evidence


def test_cause_consequence_path_respects_hop_limit() -> None:
    edges = (
        _typed_edge(
            "Blocked inflow line",
            "causes_deviation",
            "Low level",
            EntityType.CAUSE,
            EntityType.DEVIATION,
            "row-1-cause",
        ),
        _typed_edge(
            "Low level",
            "leads_to",
            "Pump cavitation",
            EntityType.DEVIATION,
            EntityType.EFFECT,
            "row-1-effect",
        ),
        _typed_edge(
            "Pump cavitation",
            "leads_to_consequence",
            "Pump running dry",
            EntityType.EFFECT,
            EntityType.CONSEQUENCE,
            "row-1-consequence",
        ),
    )

    result = GraphReasoner(max_hops=2).reason(
        _cause_consequence_context(*edges)
    )

    assert result.status == ReasoningStatus.ABSTAINED
