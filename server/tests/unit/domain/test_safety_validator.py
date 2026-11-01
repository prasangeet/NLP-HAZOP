"""Tests for the ontology and source-evidence validation gate."""
from app.domain.entities.extracted_report import ExtractedRelationship, ExtractedReport, ExtractedScenario
from app.domain.ontology.schema import EntityType
from app.domain.services.safety_validator import HazopSafetyValidator

def test_validator_accepts_well_typed_source_quoted_relationship():
    report = ExtractedReport(
        title="Report",
        scenarios=(ExtractedScenario(
            scenario_id="S001", equipment="P-101", deviation="No flow",
            causes=("Pump failure",),
            relationships=(ExtractedRelationship(
                subject="Pump failure", predicate="causes_deviation", object="No flow",
                subject_type=EntityType.CAUSE, object_type=EntityType.DEVIATION,
                evidence_quote="Pump failure causes no flow.",
            ),),
        ),),
    )
    result = HazopSafetyValidator().validate_extracted_report(
        report, "Pump failure causes no flow."
    )
    assert result.valid
    assert not result.errors

def test_validator_rejects_bad_types_and_non_verbatim_evidence():
    report = ExtractedReport(
        title="Report",
        scenarios=(ExtractedScenario(
            scenario_id="S001", equipment="P-101", deviation="No flow",
            causes=("Other cause",),
            relationships=(ExtractedRelationship(
                subject="Pump failure", predicate="causes_deviation", object="No flow",
                subject_type=EntityType.EQUIPMENT, object_type=EntityType.CONSEQUENCE,
                evidence_quote="The pump broke.",
            ),),
        ),),
    )
    result = HazopSafetyValidator().validate_extracted_report(
        report, "Pump failure causes no flow."
    )
    codes = {item.code for item in result.errors}
    assert not result.valid
    assert "ontology_type_mismatch" in codes
    assert "scenario_field_mismatch" in codes
    assert "source_quote_not_found" in codes


def test_recovery_proposal_must_match_query_and_source_record():
    from app.domain.entities.hazop_record import CausalRelation, HazopRecord, RecordType, RelationshipOrigin
    from app.domain.ontology.schema import infer_entity_types

    record = HazopRecord(
        scenario_id="S001",
        dataset_id="study",
        equipment="P-101",
        deviation="No feed flow",
        causes=("Pump failure",),
        consequences=("Reduced feed supply",),
        record_type=RecordType.SOURCE_RECONSTRUCTED,
    )
    subject_type, object_type = infer_entity_types(
        "co_listed_with_consequence", "Pump failure", "Reduced feed supply"
    )
    proposal = CausalRelation(
        subject="Pump failure",
        predicate="co_listed_with_consequence",
        object="Reduced feed supply",
        origin=RelationshipOrigin.MODEL_INFERRED,
        evidence="The scenario context supports a feed-supply consequence.",
        dataset_id="study",
        subject_type=subject_type,
        object_type=object_type,
    )

    result = HazopSafetyValidator().validate_recovery_proposal(
        proposal,
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        dataset_id="study",
        record=record,
    )

    assert result.valid
    assert not result.errors
    assert any(issue.code == "requires_expert_review" for issue in result.warnings)


def test_recovery_proposal_rejects_candidate_outside_record_fields():
    from app.domain.entities.hazop_record import CausalRelation, HazopRecord, RecordType, RelationshipOrigin
    from app.domain.ontology.schema import infer_entity_types

    record = HazopRecord(
        scenario_id="S001",
        dataset_id="study",
        equipment="P-101",
        deviation="No feed flow",
        causes=("Pump failure",),
        consequences=("Reduced feed supply",),
        record_type=RecordType.SOURCE_RECONSTRUCTED,
    )
    subject_type, object_type = infer_entity_types(
        "co_listed_with_consequence", "Pump failure", "Fire"
    )
    proposal = CausalRelation(
        subject="Pump failure",
        predicate="co_listed_with_consequence",
        object="Fire",
        origin=RelationshipOrigin.MODEL_INFERRED,
        evidence="Model rationale.",
        dataset_id="study",
        subject_type=subject_type,
        object_type=object_type,
    )

    result = HazopSafetyValidator().validate_recovery_proposal(
        proposal,
        predicate="co_listed_with_consequence",
        known_endpoint="Pump failure",
        missing_endpoint="object",
        dataset_id="study",
        record=record,
    )

    assert not result.valid
    assert "record_constraint_violation" in {issue.code for issue in result.errors}
