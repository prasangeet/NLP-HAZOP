"""API tests for report extraction through the provider factory."""

from fastapi.testclient import TestClient

from app.api.dependencies import get_ai_provider_factory, get_graph_repository
from app.domain.entities.extracted_report import (
    ExtractedRelationship,
    ExtractedReport,
    ExtractedScenario,
)
from app.domain.ontology.schema import EntityType
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory
from app.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from app.main import create_app

SOURCE = "A pump failure causes loss of feed flow."


class FakeReportExtractor:
    def extract(self, text: str, *, title: str) -> ExtractedReport:
        assert text == SOURCE
        return ExtractedReport(
            title=title,
            scenarios=(
                ExtractedScenario(
                    scenario_id="S-001",
                    equipment="P-101",
                    deviation="No feed flow",
                    causes=("Pump failure",),
                    relationships=(
                        ExtractedRelationship(
                            subject="Pump failure",
                            predicate="causes_deviation",
                            object="No feed flow",
                            subject_type=EntityType.CAUSE,
                            object_type=EntityType.DEVIATION,
                            evidence_quote=SOURCE,
                            source_location="paragraph 1",
                        ),
                    ),
                ),
            ),
        )


class FakeAIProviderFactory(AIProviderFactory):
    def create_report_extractor(self):
        return FakeReportExtractor()

    def create_missing_link_reasoner(self):
        raise NotImplementedError


def build_client():
    app = create_app()
    repository = InMemoryGraphRepository()
    app.dependency_overrides[get_graph_repository] = lambda: repository
    app.dependency_overrides[get_ai_provider_factory] = lambda: FakeAIProviderFactory()
    return TestClient(app), repository


def test_report_extraction_validates_and_persists_source_grounded_records():
    client, repository = build_client()
    with client:
        response = client.post(
            "/api/v1/reports/extract",
            files={"file": ("pump-report.txt", SOURCE.encode(), "text/plain")},
            data={"dataset_id": "pump-report"},
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["persisted"] is True
    assert payload["scenario_count"] == 1
    assert payload["relationship_count"] == 1
    assert payload["validation"]["valid"] is True

    record = repository.get_record("S-001", dataset_id="pump-report")
    assert record is not None
    relation = record.causal_relations[0]
    assert relation.origin.value == "llm_extracted"
    assert relation.subject_type == EntityType.CAUSE
    assert relation.object_type == EntityType.DEVIATION
    assert relation.evidence == SOURCE
    assert record.validation_status.value == "unreviewed"
    assert relation.is_expert_verified is False


def test_report_extraction_rejects_unsupported_file_type():
    client, _ = build_client()
    with client:
        response = client.post(
            "/api/v1/reports/extract",
            files={"file": ("report.pdf", b"%PDF", "application/pdf")},
            data={"dataset_id": "pump-report"},
        )

    assert response.status_code == 415


def test_report_extraction_rejects_non_verbatim_extraction():
    class BadQuoteExtractor(FakeReportExtractor):
        def extract(self, text: str, *, title: str) -> ExtractedReport:
            report = super().extract(text, title=title)
            scenario = report.scenarios[0]
            relation = scenario.relationships[0]
            bad = ExtractedRelationship(
                subject=relation.subject,
                predicate=relation.predicate,
                object=relation.object,
                subject_type=relation.subject_type,
                object_type=relation.object_type,
                evidence_quote="This quote is not in the source.",
            )
            return ExtractedReport(
                title=report.title,
                scenarios=(
                    ExtractedScenario(
                        scenario_id=scenario.scenario_id,
                        equipment=scenario.equipment,
                        deviation=scenario.deviation,
                        causes=scenario.causes,
                        relationships=(bad,),
                    ),
                ),
            )

    class BadAIProviderFactory(FakeAIProviderFactory):
        def create_report_extractor(self):
            return BadQuoteExtractor()

    app = create_app()
    app.dependency_overrides[get_graph_repository] = lambda: InMemoryGraphRepository()
    app.dependency_overrides[get_ai_provider_factory] = lambda: BadAIProviderFactory()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/reports/extract",
            files={"file": ("pump-report.txt", SOURCE.encode(), "text/plain")},
            data={"dataset_id": "bad-report"},
        )

    assert response.status_code == 422
    assert "exact substring" in response.json()["detail"]
