"""Use case: extract source-grounded HAZOP knowledge and persist it."""
from __future__ import annotations
import hashlib
from datetime import UTC, datetime
from app.application.ports.report_extractor import ReportExtractor
from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.extracted_report import ExtractedReport
from app.domain.entities.hazop_record import CausalRelation, HazopRecord, RecordType, RelationshipOrigin, SourceReference, ValidationStatus
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.services.safety_validator import HazopSafetyValidator, ValidationReport
class ExtractAndStoreReport:
    """Coordinate extraction, deterministic validation and atomic repository import."""
    def __init__(self, extractor: ReportExtractor, repository: DatasetRepository, validator: HazopSafetyValidator):
        self._extractor, self._repository, self._validator = extractor, repository, validator
    def execute(self, text: str, *, title: str, dataset_id: str) -> tuple[DatasetMetadata, ValidationReport, tuple[HazopRecord, ...]]:
        if self._repository.get_dataset(dataset_id) is not None:
            raise ValueError(f"Dataset '{dataset_id}' already exists.")
        extracted = self._extractor.extract(text, title=title)
        validation = self._validator.validate_extracted_report(extracted, text)
        if not validation.valid:
            messages = "; ".join(issue.message for issue in validation.errors)
            raise ValueError(f"Extracted report failed validation: {messages}")
        records = self._to_records(extracted, dataset_id)
        metadata = DatasetMetadata(dataset_id, title, title, hashlib.sha256(text.encode("utf-8")).hexdigest(), len(records), sum(len(r.causal_relations) for r in records), datetime.now(UTC).isoformat())
        self._repository.import_dataset(metadata, records)
        return metadata, validation, records
    @staticmethod
    def _to_records(report: ExtractedReport, dataset_id: str) -> tuple[HazopRecord, ...]:
        records = []
        for index, scenario in enumerate(report.scenarios, start=1):
            relations = tuple(CausalRelation(
                subject=item.subject, predicate=item.predicate, object=item.object,
                origin=RelationshipOrigin.LLM_EXTRACTED, evidence=item.evidence_quote,
                source_id=f"{dataset_id}:{scenario.scenario_id}:extracted:{edge_index}",
                dataset_id=dataset_id, subject_type=item.subject_type, object_type=item.object_type,
            ) for edge_index, item in enumerate(scenario.relationships, start=1))
            records.append(HazopRecord(
                scenario_id=scenario.scenario_id, dataset_id=dataset_id,
                plant_unit=scenario.plant_unit, equipment=scenario.equipment,
                design_intent=scenario.design_intent, guideword=scenario.guideword,
                parameter=scenario.parameter, deviation=scenario.deviation,
                causes=scenario.causes, effects=scenario.effects,
                consequences=scenario.consequences, safeguards=scenario.safeguards,
                recommendations=scenario.recommendations,
                record_type=RecordType.UNSPECIFIED, validation_status=ValidationStatus.UNREVIEWED,
                source=SourceReference(report.title, dataset_id, scenario.source_location or f"scenario {scenario.scenario_id}",
                    "LLM-extracted propositions; every edge retains its verbatim source quote and requires expert review."),
                causal_relations=relations, row_number=index,
                raw_row={"source_title": report.title, "source_location": scenario.source_location},
            ))
        return tuple(records)
