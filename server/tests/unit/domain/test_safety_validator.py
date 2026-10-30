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
