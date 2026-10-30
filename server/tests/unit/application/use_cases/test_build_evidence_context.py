"""Tests for provenance-aware evidence context construction."""

from app.application.use_cases.build_evidence_context import (
    BuildEvidenceContext,
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


def make_candidate(
    *,
    subject: str = "Inlet valve failure",
    predicate: str = "causes",
    object_: str = "High tank level",
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "The source identifies a valve failure as a cause.",
    confidence: float | None = 0.95,
    source_id: str | None = "paper-row-12",
) -> RecoveryCandidate:
    """Create a recovery candidate for testing."""
    relation = CausalRelation(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=evidence,
        confidence=confidence,
        source_id=source_id,
    )

    return RecoveryCandidate(
        missing_value=object_,
        relation=relation,
    )


def make_request(
    *,
    scenario_id: str = "HAZOP-001",
    predicate: str = "causes",
    known_endpoint: str = "Inlet valve failure",
    missing_endpoint: str = "object",
) -> RecoveryRequest:
    """Create a missing-link recovery request."""
    return RecoveryRequest(
        scenario_id=scenario_id,
        predicate=predicate,
        known_endpoint=known_endpoint,
        missing_endpoint=missing_endpoint,
    )


def test_builds_context_from_recovery_candidates() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(make_candidate(),),
    )

    request = make_request()
    context = BuildEvidenceContext().execute(result, request)

    assert context.scenario_id == "HAZOP-001"
    assert context.predicate == "causes"
    assert context.known_endpoint == "Inlet valve failure"
    assert context.missing_endpoint == "object"
    assert context.status == RecoveryStatus.CANDIDATES_FOUND
    assert context.is_empty is False
    assert len(context.evidence_items) == 1

    item = context.evidence_items[0]
    assert item.subject == "Inlet valve failure"
    assert item.predicate == "causes"
    assert item.object == "High tank level"


def test_preserves_source_provenance() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(make_candidate(),),
    )

    context = BuildEvidenceContext().execute(result, make_request())
    item = context.evidence_items[0]

    assert item.origin == RelationshipOrigin.SOURCE
    assert item.evidence == (
        "The source identifies a valve failure as a cause."
    )
    assert item.confidence == 0.95
    assert item.source_id == "paper-row-12"


def test_labels_expert_relationships_correctly() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(
            make_candidate(origin=RelationshipOrigin.EXPERT),
        ),
    )

    context = BuildEvidenceContext().execute(result, make_request())
    item = context.evidence_items[0]

    assert item.origin == RelationshipOrigin.EXPERT
    assert "expert validation" in item.relevance_note


def test_does_not_claim_model_inference_is_established_evidence() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(
            make_candidate(origin=RelationshipOrigin.MODEL_INFERRED),
        ),
    )

    context = BuildEvidenceContext().execute(result, make_request())
    item = context.evidence_items[0]

    assert item.origin == RelationshipOrigin.MODEL_INFERRED
    assert "not established evidence" in item.relevance_note


def test_empty_recovery_produces_empty_context() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.NO_CANDIDATES,
        scenario_id="HAZOP-001",
    )

    context = BuildEvidenceContext().execute(result, make_request())

    assert context.is_empty
    assert context.evidence_items == ()
    assert context.status == RecoveryStatus.NO_CANDIDATES
    assert context.predicate == "causes"
    assert context.known_endpoint == "Inlet valve failure"
    assert context.missing_endpoint == "object"


def test_context_serializes_to_json_friendly_dictionary() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(make_candidate(),),
    )

    context = BuildEvidenceContext().execute(result, make_request())
    payload = context.to_dict()

    assert payload["scenario_id"] == "HAZOP-001"
    assert payload["predicate"] == "causes"
    assert payload["known_endpoint"] == "Inlet valve failure"
    assert payload["missing_endpoint"] == "object"
    assert payload["status"] == "candidates_found"
    assert payload["is_empty"] is False

    items = payload["evidence_items"]
    assert isinstance(items, list)
    assert items[0]["origin"] == "source"
    assert items[0]["source_id"] == "paper-row-12"


def test_rejects_request_for_different_scenario() -> None:
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(make_candidate(),),
    )

    request = make_request(scenario_id="HAZOP-002")

    try:
        BuildEvidenceContext().execute(result, request)
    except ValueError as exc:
        assert "scenario_id" in str(exc)
    else:
        raise AssertionError(
            "Expected a ValueError for mismatched scenario IDs."
        )

def test_unspecified_origin_builds_context_without_key_error() -> None:
    relation = CausalRelation(
        subject="Valve failure",
        predicate="causes",
        object="High level",
        origin=RelationshipOrigin.UNSPECIFIED,
        evidence="Origin unavailable in the imported dataset.",
        source_id="dataset-x:S001:cause:1",
        dataset_id="dataset-x",
    )
    candidate = RecoveryCandidate(
        missing_value="High level",
        relation=relation,
        scenario_id="S001",
    )
    request = RecoveryRequest(
        scenario_id="S001",
        dataset_id="dataset-x",
        predicate="causes",
        known_endpoint="Valve failure",
        missing_endpoint="object",
    )
    recovery = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="S001",
        dataset_id="dataset-x",
        candidates=(candidate,),
    )

    context = BuildEvidenceContext().execute(recovery, request)

    assert context.dataset_id == "dataset-x"
    assert context.evidence_items[0].origin == RelationshipOrigin.UNSPECIFIED
    assert "Origin is unspecified" in context.evidence_items[0].relevance_note


def test_evidence_builder_rejects_dataset_mismatch() -> None:
    recovery = RecoveryResult(
        status=RecoveryStatus.NO_CANDIDATES,
        scenario_id="S001",
        dataset_id="dataset-a",
    )
    request = RecoveryRequest(
        scenario_id="S001",
        dataset_id="dataset-b",
        predicate="causes",
        known_endpoint="Valve failure",
        missing_endpoint="object",
    )
    try:
        BuildEvidenceContext().execute(recovery, request)
    except ValueError as exc:
        assert "dataset_id" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched dataset IDs.")

def test_vector_candidates_are_not_promoted_to_graph_context() -> None:
    """Retrieval candidates and traversable graph edges remain separate."""
    candidate = make_candidate()
    result = RecoveryResult(
        status=RecoveryStatus.CANDIDATES_FOUND,
        scenario_id="HAZOP-001",
        candidates=(candidate,),
        graph_context=(),
    )

    context = BuildEvidenceContext().execute(result, make_request())

    assert len(context.evidence_items) == 1
    assert context.graph_context == ()
