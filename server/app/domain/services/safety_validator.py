"""Deterministic ontology, field-consistency and evidence validation."""
from __future__ import annotations
from dataclasses import dataclass
from app.domain.entities.extracted_report import ExtractedReport, ExtractedScenario
from app.domain.entities.hazop_record import CausalRelation, HazopRecord, RelationshipOrigin
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


    def validate_recovery_proposal(
        self,
        relation: CausalRelation,
        *,
        predicate: str,
        known_endpoint: str,
        missing_endpoint: str,
        dataset_id: str,
        record: HazopRecord,
    ) -> ValidationReport:
        """Validate a proposed edge against the query and its source row.

        These are deterministic structural/record-consistency constraints,
        not a physical-process simulation or expert safety assessment.
        """
        base = self.validate_proposal(relation)
        errors = list(base.errors)
        warnings = list(base.warnings)

        if relation.predicate != predicate:
            errors.append(ValidationIssue(
                "query_predicate_mismatch",
                "The proposed predicate differs from the requested predicate.",
            ))
        if relation.dataset_id != dataset_id or record.dataset_id != dataset_id:
            errors.append(ValidationIssue(
                "dataset_scope_mismatch",
                "The proposal, query and source record must belong to the same dataset.",
            ))

        if missing_endpoint == "object":
            aligned = relation.subject.strip().casefold() == known_endpoint.strip().casefold()
            missing_value = relation.object
        elif missing_endpoint == "subject":
            aligned = relation.object.strip().casefold() == known_endpoint.strip().casefold()
            missing_value = relation.subject
        else:
            aligned = False
            missing_value = ""

        if not aligned:
            errors.append(ValidationIssue(
                "query_endpoint_mismatch",
                "The proposal does not preserve the known endpoint and requested direction.",
            ))

        folded = lambda values: {value.strip().casefold() for value in values if value.strip()}
        candidate = missing_value.strip().casefold()
        cause_values = folded(record.causes)
        effect_values = folded(record.effects)
        consequence_values = folded(record.consequences)
        safeguard_values = folded(record.safeguards)
        recommendation_values = folded(record.recommendations)
        deviation = record.deviation.strip().casefold()
        equipment = record.equipment.strip().casefold()
        unit = record.plant_unit.strip().casefold()
        parameter = record.parameter.strip().casefold()

        expected = {
            "causes_deviation": (
                relation.subject.strip().casefold() in cause_values
                and relation.object.strip().casefold() == deviation
            ),
            "leads_to": (
                relation.subject.strip().casefold() == deviation
                and relation.object.strip().casefold() in effect_values
            ),
            "leads_to_consequence": (
                relation.subject.strip().casefold() in (effect_values | {deviation})
                and relation.object.strip().casefold() in consequence_values
            ),
            "co_occurs_with_consequence": (
                relation.subject.strip().casefold() == deviation
                and relation.object.strip().casefold() in consequence_values
            ),
            "co_listed_with_consequence": (
                relation.subject.strip().casefold() in cause_values
                and relation.object.strip().casefold() in consequence_values
            ),
            "mitigated_by": (
                relation.subject.strip().casefold() in (effect_values | consequence_values | {deviation})
                and relation.object.strip().casefold() in safeguard_values
            ),
            "recommends_action": (
                relation.subject.strip().casefold() in (consequence_values | {deviation})
                and relation.object.strip().casefold() in recommendation_values
            ),
            "has_parameter": (
                relation.subject.strip().casefold() == equipment
                and bool(parameter)
                and relation.object.strip().casefold() == parameter
            ),
            "located_in": (
                relation.subject.strip().casefold() == equipment
                and bool(unit)
                and relation.object.strip().casefold() == unit
            ),
        }
        if relation.predicate not in expected:
            errors.append(ValidationIssue(
                "unsupported_process_constraint",
                f"No record-consistency constraint is defined for {relation.predicate!r}.",
            ))
        elif not expected[relation.predicate]:
            errors.append(ValidationIssue(
                "record_constraint_violation",
                "The proposed endpoints do not match the corresponding fields in the selected HAZOP record.",
            ))

        return ValidationReport(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )
