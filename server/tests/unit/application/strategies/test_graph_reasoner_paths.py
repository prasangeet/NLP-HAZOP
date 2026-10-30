"""Tests for provenance-aware multi-hop graph reasoning."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


def edge(
    subject: str,
    predicate: str,
    object_: str,
    *,
    scenario_id: str,
    origin: RelationshipOrigin = RelationshipOrigin.RECONSTRUCTED,
) -> EvidenceItem:
    return EvidenceItem(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=f"Row-associated edge from {scenario_id}.",
        confidence=None,
        source_id=f"{scenario_id}:{predicate}",
        relevance_note="Test graph edge.",
        scenario_id=scenario_id,
        scope=RetrievalScope.CROSS_SCENARIO,
    )


def graph_context(
    edges: tuple[EvidenceItem, ...],
    *,
    known: str = "Inlet valve fails open",
    missing_endpoint: str = "object",
) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="causes",
        known_endpoint=known,
        missing_endpoint=missing_endpoint,
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        graph_context=edges,
        scenario_context="T-301 | High level",
    )


def three_hop_edges() -> tuple[EvidenceItem, ...]:
    return (
        edge(
            "Inlet valve fails open",
            "causes_deviation",
            "High tank level",
            scenario_id="TRAIN-001",
        ),
        edge(
            "High tank level",
            "leads_to",
            "Tank overfill",
            scenario_id="TRAIN-002",
        ),
        edge(
            "Tank overfill",
            "leads_to_consequence",
            "Loss of containment",
            scenario_id="TRAIN-003",
        ),
    )


def test_recovers_missing_object_through_three_hop_path() -> None:
    result = GraphReasoner(max_hops=4).reason(
        graph_context(three_hop_edges())
    )

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Inlet valve fails open"
    assert result.proposed_relation.object == "Loss of containment"
    assert result.proposed_relation.origin == RelationshipOrigin.GRAPH_INFERRED
    assert result.proposed_relation.confidence is None
    assert len(result.supporting_evidence) == 1
    path = result.supporting_evidence[0]
    assert "TRAIN-001" in path
    assert "TRAIN-002" in path
    assert "TRAIN-003" in path


def test_recovers_missing_subject_by_traversing_path_in_reverse() -> None:
    result = GraphReasoner(max_hops=4).reason(
        graph_context(
            three_hop_edges(),
            known="Loss of containment",
            missing_endpoint="subject",
        )
    )

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.subject == "Inlet valve fails open"
    assert result.proposed_relation.object == "Loss of containment"


def test_abstains_when_multiple_terminal_nodes_are_reachable() -> None:
    edges = (
        edge(
            "Inlet valve fails open",
            "causes_deviation",
            "High tank level",
            scenario_id="TRAIN-001",
        ),
        edge(
            "High tank level",
            "leads_to_consequence",
            "Tank overflow",
            scenario_id="TRAIN-001",
        ),
        edge(
            "High tank level",
            "leads_to_consequence",
            "Loss of containment",
            scenario_id="TRAIN-002",
        ),
    )

    result = GraphReasoner().reason(graph_context(edges))

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.AMBIGUOUS_EVIDENCE
    assert len(result.supporting_evidence) == 2


def test_ignores_model_inferred_edges_in_graph_context() -> None:
    edges = (
        edge(
            "Inlet valve fails open",
            "causes_deviation",
            "High tank level",
            scenario_id="MODEL-001",
            origin=RelationshipOrigin.MODEL_INFERRED,
        ),
        edge(
            "High tank level",
            "leads_to_consequence",
            "Tank overflow",
            scenario_id="TRAIN-001",
        ),
    )

    result = GraphReasoner().reason(graph_context(edges))

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_does_not_compose_non_causal_association_edges() -> None:
    edges = (
        edge(
            "Inlet valve fails open",
            "associated_with",
            "High tank level",
            scenario_id="TRAIN-001",
        ),
        edge(
            "High tank level",
            "leads_to_consequence",
            "Tank overflow",
            scenario_id="TRAIN-001",
        ),
    )

    result = GraphReasoner().reason(graph_context(edges))

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT


def test_max_hops_bounds_graph_search() -> None:
    result = GraphReasoner(max_hops=2).reason(
        graph_context(three_hop_edges())
    )

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT
