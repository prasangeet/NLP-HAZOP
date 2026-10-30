"""Tests for the HAZOP graph repository contract."""

from collections.abc import Sequence

import pytest

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
)
from app.domain.repositories.graph_repository import GraphRepository


class InMemoryGraphRepository:
    """In-memory implementation used to test the repository contract."""

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
        if limit < 1:
            raise ValueError("limit must be greater than zero")

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
        if limit < 1:
            raise ValueError("limit must be greater than zero")

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
def repository() -> InMemoryGraphRepository:
    return InMemoryGraphRepository()


@pytest.fixture
def sample_record() -> HazopRecord:
    return HazopRecord(
        scenario_id="HAZOP-001",
        plant_unit="Storage",
        equipment="T-301",
        parameter="Level",
        guideword="More",
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
        evidence="The source identifies valve failure as a cause.",
        confidence=1.0,
        source_id="HAZOP-001",
    )


def test_implementation_satisfies_repository_protocol(
    repository: InMemoryGraphRepository,
) -> None:
    assert isinstance(repository, GraphRepository)


def test_save_and_get_record(
    repository: InMemoryGraphRepository,
    sample_record: HazopRecord,
) -> None:
    repository.save_record(sample_record)

    retrieved = repository.get_record("HAZOP-001")

    assert retrieved == sample_record


def test_get_missing_record_returns_none(
    repository: InMemoryGraphRepository,
) -> None:
    assert repository.get_record("UNKNOWN") is None


def test_saving_existing_scenario_id_updates_record(
    repository: InMemoryGraphRepository,
    sample_record: HazopRecord,
) -> None:
    repository.save_record(sample_record)

    updated_record = HazopRecord(
        scenario_id="HAZOP-001",
        equipment="T-301",
        deviation="Low level",
    )
    repository.save_record(updated_record)

    assert repository.get_record("HAZOP-001") == updated_record


def test_find_records_by_equipment(
    repository: InMemoryGraphRepository,
    sample_record: HazopRecord,
) -> None:
    repository.save_record(sample_record)
    repository.save_record(
        HazopRecord(
            scenario_id="HAZOP-002",
            equipment="P-101",
            deviation="Low flow",
        )
    )

    results = repository.find_records(equipment="t-301")

    assert len(results) == 1
    assert results[0].scenario_id == "HAZOP-001"


def test_find_records_by_deviation(
    repository: InMemoryGraphRepository,
    sample_record: HazopRecord,
) -> None:
    repository.save_record(sample_record)

    results = repository.find_records(deviation="high LEVEL")

    assert len(results) == 1
    assert results[0].scenario_id == "HAZOP-001"


def test_find_records_applies_multiple_filters(
    repository: InMemoryGraphRepository,
    sample_record: HazopRecord,
) -> None:
    repository.save_record(sample_record)
    repository.save_record(
        HazopRecord(
            scenario_id="HAZOP-002",
            equipment="T-301",
            deviation="Low level",
        )
    )

    results = repository.find_records(
        equipment="T-301",
        deviation="High level",
    )

    assert [record.scenario_id for record in results] == ["HAZOP-001"]


def test_find_records_respects_limit(
    repository: InMemoryGraphRepository,
) -> None:
    for index in range(3):
        repository.save_record(
            HazopRecord(
                scenario_id=f"HAZOP-{index:03}",
                equipment="T-301",
                deviation="High level",
            )
        )

    results = repository.find_records(limit=2)

    assert len(results) == 2


def test_save_and_find_relations(
    repository: InMemoryGraphRepository,
    sample_relation: CausalRelation,
) -> None:
    repository.save_relations([sample_relation])

    results = repository.find_relations(
        subject="inlet valve fails open",
        predicate="CAUSES",
    )

    assert results == [sample_relation]


def test_find_relations_by_object(
    repository: InMemoryGraphRepository,
    sample_relation: CausalRelation,
) -> None:
    repository.save_relations([sample_relation])

    results = repository.find_relations(object="High tank level")

    assert results == [sample_relation]


def test_find_relations_respects_limit(
    repository: InMemoryGraphRepository,
    sample_relation: CausalRelation,
) -> None:
    repository.save_relations(
        [sample_relation, sample_relation, sample_relation]
    )

    results = repository.find_relations(limit=2)

    assert len(results) == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_find_records_rejects_invalid_limit(
    repository: InMemoryGraphRepository,
    limit: int,
) -> None:
    with pytest.raises(ValueError):
        repository.find_records(limit=limit)


@pytest.mark.parametrize("limit", [0, -1])
def test_find_relations_rejects_invalid_limit(
    repository: InMemoryGraphRepository,
    limit: int,
) -> None:
    with pytest.raises(ValueError):
        repository.find_relations(limit=limit)
