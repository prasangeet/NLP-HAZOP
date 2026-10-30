"""Tests for the sparse TF-IDF HAZOP retriever."""

from app.application.strategies.retrieval.models import RetrievalRequest
from app.application.strategies.retrieval.tfidf import TfidfRetriever
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
    SourceReference,
)
from app.infrastructure.graph.in_memory_graph_repository import (
    InMemoryGraphRepository,
)


def record(
    scenario_id: str,
    *,
    causes: tuple[str, ...],
    consequences: tuple[str, ...],
    deviation: str = "High tank level",
    unit: str = "Storage",
) -> HazopRecord:
    return HazopRecord(
        scenario_id=scenario_id,
        plant_unit=unit,
        equipment="T-301",
        deviation=deviation,
        causes=causes,
        consequences=consequences,
        record_type=RecordType.SOURCE_RECONSTRUCTED,
        source=SourceReference(
            title="Reconstructed source row",
            document_id="single-2020-t301",
            location=scenario_id,
            provenance_note="Reconstructed; not a verbatim worksheet row.",
        ),
        causal_relations=(
            CausalRelation(
                subject=causes[0],
                predicate="causes_deviation",
                object=deviation,
                origin=RelationshipOrigin.RECONSTRUCTED,
                evidence="Cause and deviation co-listed in row.",
                source_id=f"{scenario_id}:cause",
            ),
            CausalRelation(
                subject=deviation,
                predicate="leads_to_consequence",
                object=consequences[0],
                origin=RelationshipOrigin.RECONSTRUCTED,
                evidence="Deviation and consequence co-listed in row.",
                source_id=f"{scenario_id}:consequence",
            ),
        ),
    )


def test_retrieves_candidate_and_marks_it_retrieval_inferred() -> None:
    repository = InMemoryGraphRepository()
    repository.save_record(
        record(
            "TRAIN-001",
            causes=("Inlet valve fails open",),
            consequences=("Tank overflow", "Loss of containment"),
        )
    )
    repository.save_context_record(
        HazopRecord(
            scenario_id="HELD-OUT-001",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High level",
            record_type=RecordType.SOURCE_RECONSTRUCTED,
            source=SourceReference(
                title="Held-out target",
                document_id="single-2020-t301",
            ),
        )
    )

    retriever = TfidfRetriever(repository)
    result = retriever.retrieve(
        RetrievalRequest(
            scenario_id="HELD-OUT-001",
            predicate="causes",
            known_endpoint="Inlet valve fails open",
            missing_endpoint="object",
            limit=10,
        )
    )

    assert result.scenario_exists
    assert all(
        record.scenario_id != "HELD-OUT-001"
        for record in repository.find_records(limit=100)
    )
    assert result.evidence
    first = result.evidence[0]
    assert first.relation.subject == "Inlet valve fails open"
    assert first.relation.object in {"Tank overflow", "Loss of containment"}
    assert first.relation.origin == RelationshipOrigin.RETRIEVAL_INFERRED
    assert first.relevance_score is not None
    assert first.relevance_score > 0.0
    assert "co-occurrence-based retrieval" in first.relation.evidence
    assert first.relation.source_id == "single-2020-t301:TRAIN-001"


def test_does_not_return_unrelated_rows_from_metadata_overlap() -> None:
    repository = InMemoryGraphRepository()
    repository.save_record(
        record(
            "TRAIN-001",
            causes=("Blocked outlet",),
            consequences=("Tank overflow",),
        )
    )
    repository.save_context_record(
        HazopRecord(
            scenario_id="HELD-OUT-001",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High tank level",
        )
    )

    result = TfidfRetriever(repository).retrieve(
        RetrievalRequest(
            scenario_id="HELD-OUT-001",
            predicate="causes",
            known_endpoint="A completely unrelated phrase",
            missing_endpoint="object",
        )
    )

    assert result.evidence == ()


def test_missing_target_scenario_is_reported() -> None:
    repository = InMemoryGraphRepository()
    result = TfidfRetriever(repository).retrieve(
        RetrievalRequest(
            scenario_id="UNKNOWN",
            predicate="causes",
            known_endpoint="Valve failure",
            missing_endpoint="object",
        )
    )
    assert not result.scenario_exists
    assert result.evidence == ()


def test_index_can_be_refreshed_after_repository_updates() -> None:
    repository = InMemoryGraphRepository()
    repository.save_record(
        HazopRecord(
            scenario_id="HELD-OUT-001",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High level",
        )
    )
    retriever = TfidfRetriever(repository)
    initial_build_ms = retriever.index_build_ms

    repository.save_record(
        record(
            "TRAIN-001",
            causes=("Valve fails open",),
            consequences=("Tank overflow",),
        )
    )
    retriever.refresh()

    result = retriever.retrieve(
        RetrievalRequest(
            scenario_id="HELD-OUT-001",
            predicate="causes",
            known_endpoint="Valve fails open",
            missing_endpoint="object",
        )
    )
    assert retriever.index_build_ms >= 0.0
    assert result.evidence
