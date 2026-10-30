"""Deterministic validation for extracted and inferred HAZOP relationships."""
from __future__ import annotations
from dataclasses import dataclass
from app.domain.entities.extracted_report import ExtractedReport
from app.domain.entities.hazop_record import CausalRelation, RelationshipOrigin
from app.domain.ontology.schema import validate_relation_types

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    source_location: str = ""

@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    def to_dict(self) -> dict[str, object]:
        def serialize(items):
            return [{"code": i.code, "message": i.message, "source_location": i.source_location} for i in items]
        return {"valid": self.valid, "errors": serialize(self.errors), "warnings": serialize(self.warnings)}

class HazopSafetyValidator:
    """Structural/evidence checks; not a physics simulator or expert reviewer."""
    def validate_extracted_report(self, report: ExtractedReport, source_text: str) -> ValidationReport:
        errors = []
        seen = set()
        for scenario in report.scenarios:
            key = scenario.scenario_id.casefold()
            if key in seen:
                errors.append(ValidationIssue("duplicate_scenario", f"Duplicate scenario ID: {scenario.scenario_id}", scenario.source_location))
            seen.add(key)
            for relation in scenario.relationships:
                violation = validate_relation_types(relation.predicate, relation.subject_type, relation.object_type)
                if violation:
                    errors.append(ValidationIssue("ontology_type_mismatch", violation, relation.source_location or scenario.source_location))
                if relation.evidence_quote not in source_text:
                    errors.append(ValidationIssue("source_quote_not_found", "The extracted evidence quote is not an exact substring of the supplied report.", relation.source_location or scenario.source_location))
        return ValidationReport(valid=not errors, errors=tuple(errors))

    def validate_proposal(self, relation: CausalRelation) -> ValidationReport:
        errors = []
        warnings = []
        violation = validate_relation_types(relation.predicate, relation.subject_type, relation.object_type)
        if violation:
            errors.append(ValidationIssue("ontology_type_mismatch", violation))
        if not relation.evidence.strip():
            errors.append(ValidationIssue("missing_evidence", "A recovery proposal must include supporting evidence."))
        if relation.is_expert_verified and relation.origin != RelationshipOrigin.EXPERT:
            errors.append(ValidationIssue("invalid_expert_status", "Only explicitly expert-origin relationships can be expert-verified."))
        if relation.origin in {
            RelationshipOrigin.MODEL_INFERRED, RelationshipOrigin.RETRIEVAL_INFERRED,
            RelationshipOrigin.GRAPH_INFERRED, RelationshipOrigin.LLM_EXTRACTED,
            RelationshipOrigin.SYNTHETIC, RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.UNSPECIFIED,
        }:
            warnings.append(ValidationIssue("requires_expert_review", f"Proposal origin is {relation.origin.value}; it must not be treated as expert-validated."))
        return ValidationReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))
