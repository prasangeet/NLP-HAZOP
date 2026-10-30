"""Tests for hybrid graph-vector-LLM reasoning orchestration."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import RelationshipOrigin
from app.domain.ontology.schema import infer_entity_types
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStatus,
)


class CapturingReasoner:
    def __init__(self) -> None:
        self.context = None

    def reason(self, context) -> ReasoningResult:
        self.context = context
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=context.scenario_id,
            abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
        )


def make_item(
    subject: str,
    predicate: str,
    object_: str,
    origin: RelationshipOrigin,
    source_id: str,
) -> EvidenceItem:
    return EvidenceItem(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=f"Evidence for {source_id}.",
        confidence=None,
        source_id=source_id,
        relevance_note="Test evidence.",
        scenario_id="TRAIN-001",
        scope=RetrievalScope.CROSS_SCENARIO,
        subject_type=infer_entity_types(predicate, subject, object_)[0],
        object_type=infer_entity_types(predicate, subject, object_)[1],
    )


def test_graph_step_adds_path_candidate_to_vector_evidence_before_llm() -> None:
    vector_candidate = make_item(
        "High tank level",
        "leads_to_consequence",
        "Unsupported retrieved consequence",
        RelationshipOrigin.RETRIEVAL_INFERRED,
        "vector-row-1",
    )
    graph_edges = (
        make_item(
            "High tank level",
            "leads_to",
            "Tank overfill",
            RelationshipOrigin.RECONSTRUCTED,
            "graph-row-1",
        ),
        make_item(
            "Tank overfill",
            "leads_to_consequence",
            "Loss of containment",
            RelationshipOrigin.RECONSTRUCTED,
            "graph-row-2",
        ),
    )
    context = EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="leads_to_consequence",
        known_endpoint="High tank level",
        missing_endpoint="object",
        status=RecoveryStatus.CANDIDATES_FOUND,
        evidence_items=(vector_candidate,),
        graph_context=graph_edges,
        scenario_context="Storage | T-301 | High level",
    )
    adjudicator = CapturingReasoner()

    result = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=adjudicator,
    ).reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert adjudicator.context is not None
    origins = {
        evidence.origin for evidence in adjudicator.context.evidence_items
    }
    assert RelationshipOrigin.RETRIEVAL_INFERRED in origins
    assert RelationshipOrigin.GRAPH_INFERRED in origins
    graph_item = next(
        evidence
        for evidence in adjudicator.context.evidence_items
        if evidence.origin == RelationshipOrigin.GRAPH_INFERRED
    )
    assert "graph-row-1" in graph_item.evidence
    assert "graph-row-2" in graph_item.evidence


def test_returns_graph_abstention_when_neither_retrieval_nor_path_exists() -> None:
    context = EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="causes",
        known_endpoint="Unknown cause",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        graph_context=(),
        scenario_context="Storage | T-301",
    )
    adjudicator = CapturingReasoner()

    result = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=adjudicator,
    ).reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE
    assert adjudicator.context is None
