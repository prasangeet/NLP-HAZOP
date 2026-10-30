"""Repository interface for HAZOP graph operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
)


@runtime_checkable
class GraphRepository(Protocol):
    """Contract for persisting and querying HAZOP records and relations."""

    def save_record(self, record: HazopRecord) -> None:
        """Persist a HAZOP record and its associated graph data."""
        ...

    def get_record(self, scenario_id: str) -> HazopRecord | None:
        """Retrieve a record by its unique scenario ID."""
        ...

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        """Find records matching the supplied filters."""
        ...

    def save_relations(
        self,
        relations: Sequence[CausalRelation],
    ) -> None:
        """Persist causal or semantic relationships."""
        ...

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        """Query relationships using optional filters."""
        ...
