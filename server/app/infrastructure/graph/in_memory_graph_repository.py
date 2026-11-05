"""In-memory graph repository for reproducible experiments and tests."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.hazop_record import CausalRelation, HazopRecord


class InMemoryGraphRepository:
    """Deterministic repository adapter for tests and local benchmarks."""

    storage_type = "memory"
    is_persistent = False

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetMetadata] = {}
        self._records: dict[tuple[str, str], HazopRecord] = {}
        self._context_records: dict[tuple[str, str], HazopRecord] = {}

        # Keep record-owned and standalone relationships separate so that
        # updating a record never deletes unrelated relationship evidence.
        self._record_relations: dict[
            tuple[str, str], tuple[CausalRelation, ...]
        ] = {}
        self._standalone_relations: list[CausalRelation] = []

    def import_dataset(
        self,
        dataset: DatasetMetadata,
        records: Sequence[HazopRecord],
    ) -> None:
        """Import metadata, records and relationships as one in-memory action."""
        if dataset.dataset_id in self._datasets:
            raise ValueError(f"Dataset '{dataset.dataset_id}' already exists.")

        record_map: dict[tuple[str, str], HazopRecord] = {}
        relation_map: dict[
            tuple[str, str], tuple[CausalRelation, ...]
        ] = {}

        relationship_count = 0

        for record in records:
            if record.dataset_id != dataset.dataset_id:
                raise ValueError(
                    "All records must belong to the imported dataset."
                )

            key = (record.dataset_id, record.scenario_id)

            if key in record_map:
                raise ValueError(
                    f"Duplicate scenario ID '{record.scenario_id}' in dataset."
                )

            record_map[key] = record
            relation_map[key] = record.causal_relations
            relationship_count += len(record.causal_relations)

        if len(record_map) != dataset.record_count:
            raise ValueError(
                "Dataset record_count does not match imported records."
            )

        if relationship_count != dataset.relationship_count:
            raise ValueError(
                "Dataset relationship_count does not match imported "
                "relationships."
            )

        # Commit only after all validation succeeds.
        self._datasets[dataset.dataset_id] = dataset
        self._records.update(record_map)
        self._record_relations.update(relation_map)

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        return self._datasets.get(dataset_id)

    def find_datasets(
        self,
        *,
        limit: int = 100,
    ) -> Sequence[DatasetMetadata]:
        self._validate_limit(limit)

        values = sorted(
            self._datasets.values(),
            key=lambda item: (item.created_at, item.dataset_id),
            reverse=True,
        )
        return values[:limit]

    def save_record(self, record: HazopRecord) -> None:
        """Save a record and replace only its owned relationships."""
        key = (record.dataset_id, record.scenario_id)

        self._records[key] = record
        self._record_relations[key] = record.causal_relations

    def save_context_record(self, record: HazopRecord) -> None:
        """Register metadata for a held-out target without indexing it."""
        key = (record.dataset_id, record.scenario_id)
        self._context_records[key] = record

    def get_record(
        self,
        scenario_id: str,
        *,
        dataset_id: str = "default",
    ) -> HazopRecord | None:
        if not dataset_id.strip():
            raise ValueError("dataset_id must not be empty")

        key = (dataset_id.strip(), scenario_id)
        return self._records.get(key) or self._context_records.get(key)

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        dataset_id: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        self._validate_limit(limit)

        records = list(self._records.values())

        if dataset_id is not None:
            expected_dataset = dataset_id.casefold()
            records = [
                record
                for record in records
                if record.dataset_id.casefold() == expected_dataset
            ]

        if equipment is not None:
            expected_equipment = equipment.casefold()
            records = [
                record
                for record in records
                if record.equipment.casefold() == expected_equipment
            ]

        if deviation is not None:
            expected_deviation = deviation.casefold()
            records = [
                record
                for record in records
                if record.deviation.casefold() == expected_deviation
            ]

        return records[:limit]

    def save_relations(
        self,
        relations: Sequence[CausalRelation],
    ) -> None:
        """Save standalone relationships without collapsing evidence."""
        self._standalone_relations.extend(relations)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        dataset_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        self._validate_limit(limit)

        results = list(self._standalone_relations)

        for relations in self._record_relations.values():
            results.extend(relations)

        if dataset_id is not None:
            expected_dataset = dataset_id.casefold()
            results = [
                relation
                for relation in results
                if relation.dataset_id.casefold() == expected_dataset
            ]

        if subject is not None:
            expected_subject = subject.casefold()
            results = [
                relation
                for relation in results
                if relation.subject.casefold() == expected_subject
            ]

        if predicate is not None:
            expected_predicate = predicate.casefold()
            results = [
                relation
                for relation in results
                if relation.predicate.casefold() == expected_predicate
            ]

        if object is not None:
            expected_object = object.casefold()
            results = [
                relation
                for relation in results
                if relation.object.casefold() == expected_object
            ]

        return results[:limit]

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be greater than zero")
