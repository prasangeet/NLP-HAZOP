"""Tests for hybrid graph-vector-LLM reasoning orchestration."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.application.strategies.retrieval.models import RetrievalScope
from app.application.use_cases.build_evidence_context import (
    EvidenceContext,
    EvidenceItem,
)
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.entities.hazop_record import CausalRelation, RelationshipOrigin
from app.domain.ontology.schema import infer_entity_types
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStatus,
)


class CapturingReasoner:
    def __init__(self, relations=()) -> None:
        self.context = None
        self.relations = tuple(relations)

    def reason_all(self, context):
        self.context = context
        return self.relations

    def reason(self, context) -> ReasoningResult:
        self.context = context
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=context.scenario_id,
            dataset_id=getattr(context, "dataset_id", "default"),
            abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
        )


class FixedProposalReasoner:
    def __init__(self, relation: CausalRelation) -> None:
        self.relation = relation
        self.context = None

    def reason(self, context) -> ReasoningResult:
        self.context = context
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            dataset_id=context.dataset_id,
            proposed_relation=self.relation,
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


def make_context(*items: EvidenceItem) -> EvidenceContext:
    return EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        status=(
            RecoveryStatus.CANDIDATES_FOUND
            if items
            else RecoveryStatus.NO_CANDIDATES
        ),
        evidence_items=items,
        scenario_context="P-101 | No feed flow | Pump failure",
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


def test_hybrid_uses_generative_llm_fallback_when_retrieval_is_empty() -> None:
    context = EvidenceContext(
        scenario_id="HELD-OUT-001",
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        graph_context=(),
        scenario_context="Equipment: P-101 | No feed flow | Cause: Pump failure",
    )
    fallback_relation = CausalRelation(
        subject="Pump failure",
        predicate="co_listed_with_consequence",
        object="Pump breakdown",
        origin=RelationshipOrigin.MODEL_INFERRED,
        evidence="LLM-generated hypothesis from scenario context.",
        dataset_id="default",
        subject_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[0],
        object_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[1],
    )
    fallback = FixedProposalReasoner(fallback_relation)

    result = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=CapturingReasoner(),
        fallback_reasoner=fallback,
    ).reason(context)

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.origin == RelationshipOrigin.MODEL_INFERRED
    assert fallback.context is not None
    assert fallback.context.evidence_items == ()


def test_reason_all_combines_evidence_candidates_with_distinct_model_hypothesis() -> None:
    candidate = make_item(
        "Pump failure",
        "co_listed_with_consequence",
        "Loss of containment",
        RelationshipOrigin.RETRIEVAL_INFERRED,
        "retrieval-row-1",
    )
    supported = CausalRelation(
        subject=candidate.subject,
        predicate=candidate.predicate,
        object=candidate.object,
        origin=candidate.origin,
        evidence=candidate.evidence,
        source_id=candidate.source_id,
        subject_type=candidate.subject_type,
        object_type=candidate.object_type,
    )
    generated = CausalRelation(
        subject="Pump failure",
        predicate="co_listed_with_consequence",
        object="Pump breakdown",
        origin=RelationshipOrigin.MODEL_INFERRED,
        evidence="LLM-generated hypothesis based on scenario context.",
        subject_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[0],
        object_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[1],
    )
    adjudicator = CapturingReasoner((supported,))
    fallback = FixedProposalReasoner(generated)
    context = make_context(candidate)

    relations = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=adjudicator,
        fallback_reasoner=fallback,
    ).reason_all(context)

    assert [item.object for item in relations] == [
        "Loss of containment",
        "Pump breakdown",
    ]
    assert [item.origin for item in relations] == [
        RelationshipOrigin.RETRIEVAL_INFERRED,
        RelationshipOrigin.MODEL_INFERRED,
    ]
    assert fallback.context is not None
    assert fallback.context.evidence_items == ()
    assert adjudicator.context is not None
    assert len(adjudicator.context.evidence_items) == 1


def test_reason_all_does_not_duplicate_model_hypothesis_already_in_evidence() -> None:
    candidate = make_item(
        "Pump failure",
        "co_listed_with_consequence",
        "Pump breakdown",
        RelationshipOrigin.RETRIEVAL_INFERRED,
        "retrieval-row-1",
    )
    supported = CausalRelation(
        subject=candidate.subject,
        predicate=candidate.predicate,
        object=candidate.object,
        origin=candidate.origin,
        evidence=candidate.evidence,
        source_id=candidate.source_id,
        subject_type=candidate.subject_type,
        object_type=candidate.object_type,
    )
    generated = CausalRelation(
        subject="Pump failure",
        predicate="co_listed_with_consequence",
        object="Pump breakdown",
        origin=RelationshipOrigin.MODEL_INFERRED,
        evidence="LLM-generated hypothesis based on scenario context.",
        subject_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[0],
        object_type=infer_entity_types(
            "co_listed_with_consequence", "Pump failure", "Pump breakdown"
        )[1],
    )

    relations = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=CapturingReasoner((supported,)),
        fallback_reasoner=FixedProposalReasoner(generated),
    ).reason_all(make_context(candidate))

    assert len(relations) == 1
    assert relations[0].origin == RelationshipOrigin.RETRIEVAL_INFERRED


def test_hybrid_uses_generative_fallback_when_retrieval_candidates_do_not_match_query() -> None:
    irrelevant_candidate = make_item(
        "Different cause",
        "leads_to_consequence",
        "Loss of containment",
        RelationshipOrigin.RETRIEVAL_INFERRED,
        "vector-row-irrelevant",
    )
    context = EvidenceContext(
        scenario_id="HELD-OUT-002",
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        status=RecoveryStatus.CANDIDATES_FOUND,
        evidence_items=(irrelevant_candidate,),
        graph_context=(),
        scenario_context="Equipment: P-101 | No feed flow | Cause: Pump failure",
    )
    fallback = CapturingReasoner()

    result = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=CapturingReasoner(),
        fallback_reasoner=fallback,
    ).reason(context)

    assert result.status == ReasoningStatus.ABSTAINED
    assert fallback.context is not None
    assert fallback.context.evidence_items == ()
    assert fallback.context.scenario_context == context.scenario_context


def test_reason_all_routes_matching_retrieval_candidates_to_llm_set_mode() -> None:
    candidate = make_item(
        "Pump failure",
        "co_listed_with_consequence",
        "Loss of feed",
        RelationshipOrigin.RETRIEVAL_INFERRED,
        "retrieval-row-1",
    )
    context = EvidenceContext(
        scenario_id="HELD-OUT-003",
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        status=RecoveryStatus.CANDIDATES_FOUND,
        evidence_items=(candidate,),
        graph_context=(),
        scenario_context="P-101 | No feed flow",
    )
    adjudicator = CapturingReasoner()

    relations = HybridReasoner(
        graph_reasoner=GraphReasoner(),
        llm_reasoner=adjudicator,
    ).reason_all(context)

    assert relations == ()
    assert adjudicator.context is not None
    assert len(adjudicator.context.evidence_items) == 1
    assert adjudicator.context.evidence_items[0].source_id == "retrieval-row-1"
    assert adjudicator.context.predicate == "co_listed_with_consequence"
