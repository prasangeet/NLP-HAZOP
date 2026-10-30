"""Tests for the HAZOP application service."""

from collections.abc import Sequence

import pytest

from app.application.services.hazop_service import HazopService
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
)


class FakeGraphRepository:
    """In-memory repository fake for application service tests."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str], HazopRecord] = {}
        self.relations: list[CausalRelation] = []

    def save_record(self, record: HazopRecord) -> None:
        self.records[(record.dataset_id, record.scenario_id)] = record

    def get_record(
        self, scenario_id: str, *, dataset_id: str = "default"
    ) -> HazopRecord | None:
        return self.records.get((dataset_id, scenario_id))

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        dataset_id: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        results = list(self.records.values())
        if dataset_id is not None:
            results = [record for record in results if record.dataset_id == dataset_id]

        if equipment is not None:
            results = [
                record
                for record in results
                if record.equipment.casefold() == equipment.casefold()
            ]

        if deviation is not None:
            results = [
                record
                for record in results
                if record.deviation.casefold() == deviation.casefold()
            ]

        return results[:limit]

    def save_relations(
        self,
        relations: Sequence[CausalRelation],
    ) -> None:
        self.relations.extend(relations)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        dataset_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        results = list(self.relations)
        if dataset_id is not None:
            results = [relation for relation in results if relation.dataset_id == dataset_id]

        if subject is not None:
            results = [
                relation
                for relation in results
                if relation.subject.casefold() == subject.casefold()
            ]

        if predicate is not None:
            results = [
                relation
                for relation in results
                if relation.predicate.casefold() == predicate.casefold()
            ]

        if object is not None:
            results = [
                relation
                for relation in results
                if relation.object.casefold() == object.casefold()
            ]

        return results[:limit]


@pytest.fixture
def repository() -> FakeGraphRepository:
    return FakeGraphRepository()


@pytest.fixture
def service(repository: FakeGraphRepository) -> HazopService:
    return HazopService(repository)


@pytest.fixture
def sample_record() -> HazopRecord:
    return HazopRecord(
        scenario_id="HAZOP-001",
        equipment="T-301",
        deviation="High level",
        causes=("Inlet valve fails open",),
        consequences=("Tank overflow",),
    )


@pytest.fixture
def sample_relation() -> CausalRelation:
    return CausalRelation(
        subject="Inlet valve fails open",
        predicate="causes",
        object="High tank level",
        origin=RelationshipOrigin.SOURCE,
        evidence="Cause documented in the source record.",
        confidence=1.0,
        source_id="HAZOP-001",
    )


def test_register_and_retrieve_record(
    service: HazopService,
    sample_record: HazopRecord,
) -> None:
    service.register_record(sample_record)

    assert service.get_record("HAZOP-001") == sample_record


def test_get_record_trims_scenario_id(
    service: HazopService,
    sample_record: HazopRecord,
) -> None:
    service.register_record(sample_record)

    assert service.get_record(" HAZOP-001 ") == sample_record


def test_get_missing_record_returns_none(
    service: HazopService,
) -> None:
    assert service.get_record("UNKNOWN") is None


def test_get_record_rejects_blank_scenario_id(
    service: HazopService,
) -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        service.get_record("   ")


def test_search_records_by_equipment(
    service: HazopService,
    sample_record: HazopRecord,
) -> None:
    service.register_record(sample_record)

    results = service.search_records(equipment=" t-301 ")

    assert len(results) == 1
    assert results[0] == sample_record


def test_search_records_by_deviation(
    service: HazopService,
    sample_record: HazopRecord,
) -> None:
    service.register_record(sample_record)

    results = service.search_records(deviation="High level")

    assert len(results) == 1
    assert results[0].scenario_id == "HAZOP-001"


def test_search_records_with_no_filters(
    service: HazopService,
    sample_record: HazopRecord,
) -> None:
    service.register_record(sample_record)

    results = service.search_records()

    assert results == [sample_record]


def test_search_records_rejects_invalid_limit(
    service: HazopService,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        service.search_records(limit=0)


def test_register_and_search_relations(
    service: HazopService,
    sample_relation: CausalRelation,
) -> None:
    service.register_relations([sample_relation])

    results = service.search_relations(
        subject=" Inlet valve fails open ",
        predicate="causes",
    )

    assert results == [sample_relation]


def test_search_relations_by_object(
    service: HazopService,
    sample_relation: CausalRelation,
) -> None:
    service.register_relations([sample_relation])

    results = service.search_relations(object="High tank level")

    assert results == [sample_relation]


def test_search_relations_rejects_invalid_limit(
    service: HazopService,
) -> None:
    with pytest.raises(ValueError, match="limit"):
        service.search_relations(limit=-1)


def test_register_relations_accepts_empty_sequence(
    service: HazopService,
    repository: FakeGraphRepository,
) -> None:
    service.register_relations([])

    assert repository.relations == []
