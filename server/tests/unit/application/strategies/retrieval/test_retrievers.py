"""Tests for scenario-local retrieval."""

from __future__ import annotations

import pytest

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalScope,
)
from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
)


class FakeRepository:
    """Minimal repository for retrieval strategy tests."""

    def __init__(self, records: list[HazopRecord]) -> None:
        self.records = {(record.dataset_id, record.scenario_id): record for record in records}

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
    ) -> list[HazopRecord]:
        records = list(self.records.values())
        if dataset_id is not None:
            records = [record for record in records if record.dataset_id == dataset_id]

        if equipment is not None:
            records = [
                record
                for record in records
                if record.equipment.casefold() == equipment.casefold()
            ]

        if deviation is not None:
            records = [
                record
                for record in records
                if record.deviation.casefold() == deviation.casefold()
            ]

        return records[:limit]

    def save_record(self, record: HazopRecord) -> None:
        self.records[(record.dataset_id, record.scenario_id)] = record

    def save_relations(self, relations) -> None:
        raise NotImplementedError

    def find_relations(self, **kwargs):
        raise NotImplementedError


def relation(
    subject: str,
    predicate: str,
    object_: str,
    *,
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
) -> CausalRelation:
    return CausalRelation(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence="Documented relationship.",
        source_id="test-source",
    )


def record(
    scenario_id: str,
    *relations: CausalRelation,
) -> HazopRecord:
    return HazopRecord(
        scenario_id=scenario_id,
        equipment="T-301",
        deviation="High level",
        causal_relations=tuple(relations),
    )


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository(
        [
            record(
                "HAZOP-001",
                relation(
                    "Inlet valve failure",
                    "causes",
                    "High tank level",
                ),
                relation(
                    "Inlet valve failure",
                    "causes",
                    "Tank overflow",
                    origin=RelationshipOrigin.MODEL_INFERRED,
                ),
            ),
            record(
                "HAZOP-002",
                relation(
                    "Inlet valve failure",
                    "causes",
                    "Tank overflow",
                ),
            ),
            record(
                "HAZOP-003",
                relation(
                    "Blocked outlet",
                    "causes",
                    "High tank level",
                ),
            ),
        ]
    )


def test_local_retrieval_returns_only_target_scenario(
    repository: FakeRepository,
) -> None:
    result = ScenarioLocalRetriever(repository).retrieve(
        RetrievalRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            known_endpoint="Inlet valve failure",
            missing_endpoint="object",
        )
    )

    assert result.scope == RetrievalScope.SCENARIO_LOCAL
    assert all(
        item.scenario_id == "HAZOP-001"
        for item in result.evidence
    )


def test_local_retrieval_excludes_model_inferred_relations(
    repository: FakeRepository,
) -> None:
    result = ScenarioLocalRetriever(repository).retrieve(
        RetrievalRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            known_endpoint="Inlet valve failure",
            missing_endpoint="object",
        )
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].relation.object == "High tank level"


def test_local_retrieval_returns_empty_for_unknown_scenario(
    repository: FakeRepository,
) -> None:
    result = ScenarioLocalRetriever(repository).retrieve(
        RetrievalRequest(
            scenario_id="HAZOP-MISSING",
            predicate="causes",
            known_endpoint="Inlet valve failure",
            missing_endpoint="object",
        )
    )

    assert result.is_empty


@pytest.mark.parametrize("limit", [0, -1])
def test_request_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        RetrievalRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            known_endpoint="Valve failure",
            missing_endpoint="object",
            limit=limit,
        )


def test_request_rejects_invalid_missing_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="MissingEndpoint must be either 'subject' or 'object'",
    ):
        RetrievalRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            known_endpoint="Valve failure",
            missing_endpoint="invalid",
        )
