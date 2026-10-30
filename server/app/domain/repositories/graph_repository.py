"""Repository interfaces for HAZOP graph and dataset operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.hazop_record import CausalRelation, HazopRecord


@runtime_checkable
class GraphRepository(Protocol):
    """Core graph repository contract used by retrieval and reasoning."""

    def save_record(self, record: HazopRecord) -> None:
        """Persist a HAZOP record and its associated graph data."""
        ...

    def get_record(
        self, scenario_id: str, *, dataset_id: str = "default"
    ) -> HazopRecord | None:
        """Retrieve a record by its dataset-scoped scenario identity."""
        ...

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        dataset_id: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        """Find records matching the supplied filters."""
        ...

    def save_relations(self, relations: Sequence[CausalRelation]) -> None:
        """Persist causal or semantic relationships."""
        ...

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        dataset_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        """Query relationships using optional filters."""
        ...


@runtime_checkable
class DatasetRepository(GraphRepository, Protocol):
    """Extended repository contract for durable dataset lifecycle operations."""

    storage_type: str
    is_persistent: bool

    def import_dataset(
        self,
        dataset: DatasetMetadata,
        records: Sequence[HazopRecord],
    ) -> None:
        """Atomically create a dataset and all of its records and graph edges."""
        ...

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        """Retrieve dataset metadata by its stable identifier."""
        ...

    def find_datasets(self, *, limit: int = 100) -> Sequence[DatasetMetadata]:
        """List datasets in deterministic newest-first order."""
        ...
