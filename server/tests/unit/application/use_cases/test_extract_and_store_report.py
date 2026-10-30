"""Tests for source-grounded HAZOP extraction and persistence."""
from app.application.use_cases.extract_and_store_report import ExtractAndStoreReport
from app.domain.entities.extracted_report import ExtractedRelationship, ExtractedReport, ExtractedScenario
from app.domain.ontology.schema import EntityType
from app.domain.services.safety_validator import HazopSafetyValidator
from app.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from app.domain.entities.hazop_record import RelationshipOrigin

SOURCE = "A pump failure causes loss of feed flow."

class FakeExtractor:
    def extract(self, text: str, *, title: str) -> ExtractedReport:
        return ExtractedReport(
            title=title,
            scenarios=(ExtractedScenario(
                scenario_id="S-001",
                equipment="P-101",
                deviation="No feed flow",
                causes=("Pump failure",),
                relationships=(ExtractedRelationship(
                    subject="Pump failure",
                    predicate="causes_deviation",
                    object="No feed flow",
                    subject_type=EntityType.CAUSE,
                    object_type=EntityType.DEVIATION,
                    evidence_quote="A pump failure causes loss of feed flow.",
                    source_location="paragraph 1",
                ),),
            ),),
        )

def test_extracted_report_is_validated_and_persisted_as_unreviewed_knowledge():
    repository = InMemoryGraphRepository()
    use_case = ExtractAndStoreReport(FakeExtractor(), repository, HazopSafetyValidator())
    metadata, validation, records = use_case.execute(
        SOURCE, title="Pump report", dataset_id="pump-report"
    )
    assert validation.valid
    assert metadata.record_count == 1
    assert metadata.relationship_count == 1
    relation = repository.get_record("S-001", dataset_id="pump-report").causal_relations[0]
    assert relation.origin == RelationshipOrigin.LLM_EXTRACTED
    assert relation.subject_type == EntityType.CAUSE
    assert relation.object_type == EntityType.DEVIATION
    assert relation.is_expert_verified is False

def test_duplicate_report_dataset_is_rejected():
    repository = InMemoryGraphRepository()
    use_case = ExtractAndStoreReport(FakeExtractor(), repository, HazopSafetyValidator())
    use_case.execute(SOURCE, title="Pump report", dataset_id="pump-report")
    try:
        use_case.execute(SOURCE, title="Pump report", dataset_id="pump-report")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate dataset rejection")
