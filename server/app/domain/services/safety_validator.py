"""Deterministic ontology, field-consistency and evidence validation."""
from __future__ import annotations
from dataclasses import dataclass
from app.domain.entities.extracted_report import ExtractedReport, ExtractedScenario
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
            return [{"code": item.code, "message": item.message, "source_location": item.source_location} for item in items]
        return {"valid": self.valid, "errors": serialize(self.errors), "warnings": serialize(self.warnings)}

class HazopSafetyValidator:
    """Structural process-safety checks; not a physics simulator or expert reviewer."""
    @staticmethod
    def _field_consistency_error(scenario: ExtractedScenario, relation) -> str | None:
        folded = lambda values: {value.strip().casefold() for value in values if value.strip()}
        subject, object_value = relation.subject.strip().casefold(), relation.object.strip().casefold()
        causes = folded(scenario.causes)
        effects = folded(scenario.effects)
        consequences = folded(scenario.consequences)
        safeguards = folded(scenario.safeguards)
        recommendations = folded(scenario.recommendations)
        deviation = scenario.deviation.strip().casefold()
        equipment = scenario.equipment.strip().casefold()
        parameter = scenario.parameter.strip().casefold()
        unit = scenario.plant_unit.strip().casefold()
        checks = {
            "causes_deviation": subject in causes and object_value == deviation,
            "leads_to": subject == deviation and object_value in effects,
            "leads_to_consequence": subject in (effects | {deviation}) and object_value in consequences,
            "co_occurs_with_consequence": subject == deviation and object_value in consequences,
            "mitigated_by": subject in (effects | consequences | {deviation}) and object_value in safeguards,
            "recommends_action": subject in (consequences | {deviation}) and object_value in recommendations,
            "has_parameter": subject == equipment and bool(parameter) and object_value == parameter,
            "located_in": subject == equipment and bool(unit) and object_value == unit,
        }
        if relation.predicate not in checks:
            return f"Unsupported predicate {relation.predicate!r}."
        if not checks[relation.predicate]:
            return (
                f"{relation.predicate!r} endpoints must agree with the scenario's "
                "extracted cause/deviation/effect/consequence/safeguard fields."
            )
        return None

    def validate_extracted_report(self, report: ExtractedReport, source_text: str) -> ValidationReport:
        errors: list[ValidationIssue] = []
        seen: set[str] = set()
        for scenario in report.scenarios:
            scenario_key = scenario.scenario_id.casefold()
            if scenario_key in seen:
                errors.append(ValidationIssue("duplicate_scenario", f"Duplicate scenario ID: {scenario.scenario_id}", scenario.source_location))
            seen.add(scenario_key)
            for relation in scenario.relationships:
                location = relation.source_location or scenario.source_location
                violation = validate_relation_types(relation.predicate, relation.subject_type, relation.object_type)
                if violation:
                    errors.append(ValidationIssue("ontology_type_mismatch", violation, location))
                consistency_error = self._field_consistency_error(scenario, relation)
                if consistency_error:
                    errors.append(ValidationIssue("scenario_field_mismatch", consistency_error, location))
                if relation.evidence_quote not in source_text:
                    errors.append(ValidationIssue("source_quote_not_found", "The extracted evidence quote is not an exact substring of the supplied report.", location))
        return ValidationReport(valid=not errors, errors=tuple(errors))

    def validate_proposal(self, relation: CausalRelation) -> ValidationReport:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        violation = validate_relation_types(relation.predicate, relation.subject_type, relation.object_type)
        if violation:
            errors.append(ValidationIssue("ontology_type_mismatch", violation))
        if not relation.evidence.strip():
            errors.append(ValidationIssue("missing_evidence", "A recovery proposal must include supporting evidence."))
        if relation.is_expert_verified and relation.origin != RelationshipOrigin.EXPERT:
            errors.append(ValidationIssue("invalid_expert_status", "Only explicitly expert-origin relationships can be expert-verified."))
        unverified_origins = {
            RelationshipOrigin.MODEL_INFERRED, RelationshipOrigin.RETRIEVAL_INFERRED,
            RelationshipOrigin.GRAPH_INFERRED, RelationshipOrigin.LLM_EXTRACTED,
            RelationshipOrigin.SYNTHETIC, RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.UNSPECIFIED,
        }
        if relation.origin in unverified_origins:
            warnings.append(ValidationIssue("requires_expert_review", f"Proposal origin is {relation.origin.value}; it must not be treated as expert-validated."))
        return ValidationReport(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))
