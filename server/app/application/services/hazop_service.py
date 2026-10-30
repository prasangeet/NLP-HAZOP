"""Application service for structured HAZOP operations."""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
)

from app.domain.repositories.graph_repository import GraphRepository


class HazopService:
    """Coordinate HAZOP operations through the repository interface."""

    def __init__(self, repository: GraphRepository) -> None:
        self._repository = repository

    def register_record(self, record: HazopRecord) -> None:
        """Register or update a HAZOP record."""
        self._repository.save_record(record)

    def get_record(self, scenario_id: str) -> HazopRecord | None:
        """Retrieve a HAZOP record by scenario ID."""
        scenario_id = scenario_id.strip()

        if not scenario_id:
            raise ValueError("scenario_id must not be empty.")

        return self._repository.get_record(scenario_id)

    def search_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        """Search HAZOP records by equipment and/or deviation."""
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        if equipment is not None:
            equipment = equipment.strip() or None

        if deviation is not None:
            deviation = deviation.strip() or None

        return self._repository.find_records(
            equipment=equipment,
            deviation=deviation,
            limit=limit,
        )

    def register_relations(
        self,
        relations: Sequence[CausalRelation],
    ) -> None:
        """Register causal or semantic relationships."""
        self._repository.save_relations(relations)

    def search_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        """Search causal relationships using optional filters."""
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        if subject is not None:
            subject = subject.strip() or None

        if predicate is not None:
            predicate = predicate.strip() or None

        if object is not None:
            object = object.strip() or None

        return self._repository.find_relations(
            subject=subject,
            predicate=predicate,
            object=object,
            limit=limit,
        )
