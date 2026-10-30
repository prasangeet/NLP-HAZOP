"""In-memory graph repository for reproducible experiments and tests."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.entities.hazop_record import CausalRelation, HazopRecord


class InMemoryGraphRepository:
    """Simple repository adapter with deterministic ordering.

    This implementation is intended for local benchmark runs and unit tests.
    It can be replaced by a persistent graph database adapter without
    changing application or domain contracts.
    """

    def __init__(self) -> None:
        self._records: dict[str, HazopRecord] = {}
        self._relations: list[CausalRelation] = []

    def save_record(self, record: HazopRecord) -> None:
        self._records[record.scenario_id] = record

    def get_record(self, scenario_id: str) -> HazopRecord | None:
        return self._records.get(scenario_id)

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be greater than zero")

        records = list(self._records.values())
        if equipment is not None:
            expected = equipment.casefold()
            records = [
                record for record in records
                if record.equipment.casefold() == expected
            ]
        if deviation is not None:
            expected = deviation.casefold()
            records = [
                record for record in records
                if record.deviation.casefold() == expected
            ]
        return records[:limit]

    def save_relations(self, relations: Sequence[CausalRelation]) -> None:
        self._relations.extend(relations)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be greater than zero")

        results = list(self._relations)
        if subject is not None:
            expected = subject.casefold()
            results = [
                item for item in results
                if item.subject.casefold() == expected
            ]
        if predicate is not None:
            expected = predicate.casefold()
            results = [
                item for item in results
                if item.predicate.casefold() == expected
            ]
        if object is not None:
            expected = object.casefold()
            results = [
                item for item in results
                if item.object.casefold() == expected
            ]
        return results[:limit]
