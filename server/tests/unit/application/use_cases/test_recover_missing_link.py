"""Tests for scenario-specific missing-link recovery."""

from __future__ import annotations

import pytest

from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.application.use_cases.recover_missing_link import (
    MissingEndpoint,
    RecoverMissingLink,
    RecoveryRequest,
    RecoveryStatus,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
    SourceReference,
)


class FakeGraphRepository:
    """In-memory graph repository for isolated unit tests."""

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
    ) -> list[HazopRecord]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

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

    def save_relations(
        self,
        relations: tuple[CausalRelation, ...] | list[CausalRelation],
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
    ) -> list[CausalRelation]:
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        relations = self.relations
        if dataset_id is not None:
            relations = [relation for relation in relations if relation.dataset_id == dataset_id]

        if subject is not None:
            relations = [
                relation
                for relation in relations
                if relation.subject.casefold() == subject.casefold()
            ]

        if predicate is not None:
            relations = [
                relation
                for relation in relations
                if relation.predicate.casefold() == predicate.casefold()
            ]

        if object is not None:
            relations = [
                relation
                for relation in relations
                if relation.object.casefold() == object.casefold()
            ]

        return relations[:limit]


def make_relation(
    *,
    subject: str = "Inlet valve fails open",
    predicate: str = "causes",
    object_: str = "High tank level",
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "Documented in the source.",
    source_id: str | None = "paper-row-12",
) -> CausalRelation:
    """Create a causal relationship for testing."""

    return CausalRelation(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=evidence,
        source_id=source_id,
    )


def make_record(
    scenario_id: str = "HAZOP-001",
    *,
    relations: tuple[CausalRelation, ...] = (),
) -> HazopRecord:
    """Create a HAZOP record containing its own relationships."""

    return HazopRecord(
        scenario_id=scenario_id,
        equipment="T-301",
        deviation="High level",
        source=SourceReference(
            title="HAZOP source document",
            document_id="paper-001",
        ),
        causal_relations=relations,
    )


@pytest.fixture
def repository() -> FakeGraphRepository:
    """Repository containing a HAZOP scenario and its evidence."""

    repo = FakeGraphRepository()

    relations = (
        make_relation(
            object_="High tank level",
            evidence="The source documents a high tank level.",
            source_id="source-001",
        ),
        make_relation(
            object_="Tank overflow",
            origin=RelationshipOrigin.EXPERT,
            evidence="An expert validated the overflow relationship.",
            source_id="expert-001",
        ),
        make_relation(
            object_="Unexpected shutdown",
            origin=RelationshipOrigin.MODEL_INFERRED,
            evidence="Suggested by a model.",
            source_id="model-001",
        ),
    )

    repo.save_record(make_record(relations=relations))
    return repo


@pytest.fixture
def use_case(repository: FakeGraphRepository) -> RecoverMissingLink:
    return RecoverMissingLink(ScenarioLocalRetriever(repository))


def test_recovers_missing_object(use_case: RecoverMissingLink) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert result.status == RecoveryStatus.CANDIDATES_FOUND
    assert {c.missing_value for c in result.candidates} == {
        "High tank level",
        "Tank overflow",
    }


def test_recovers_missing_subject(
    repository: FakeGraphRepository,
) -> None:
    repository.save_record(
        make_record(
            "HAZOP-002",
            relations=(
                make_relation(
                    subject="Blocked outlet",
                    object_="High tank level",
                    source_id="source-002",
                ),
            ),
        )
    )

    result = RecoverMissingLink(ScenarioLocalRetriever(repository)).execute(
        RecoveryRequest(
            scenario_id="HAZOP-002",
            predicate="causes",
            missing_endpoint=MissingEndpoint.SUBJECT,
            known_endpoint="High tank level",
        )
    )

    assert result.status == RecoveryStatus.CANDIDATES_FOUND
    assert len(result.candidates) == 1
    assert result.candidates[0].missing_value == "Blocked outlet"


def test_preserves_relationship_provenance(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    candidate = next(
        c for c in result.candidates
        if c.missing_value == "High tank level"
    )

    assert candidate.relation.origin == RelationshipOrigin.SOURCE
    assert candidate.relation.source_id == "source-001"
    assert candidate.relation.evidence == (
        "The source documents a high tank level."
    )


def test_excludes_model_inferred_relationships(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert all(
        c.relation.origin != RelationshipOrigin.MODEL_INFERRED
        for c in result.candidates
    )
    assert "Unexpected shutdown" not in {
        c.missing_value for c in result.candidates
    }


def test_returns_no_candidates_for_unknown_relationship(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Unknown equipment",
        )
    )

    assert result.status == RecoveryStatus.NO_CANDIDATES
    assert result.candidates == ()


def test_returns_scenario_not_found(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="UNKNOWN-SCENARIO",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert result.status == RecoveryStatus.SCENARIO_NOT_FOUND
    assert result.candidates == ()


@pytest.mark.parametrize(
    ("scenario_id", "predicate", "known_endpoint"),
    [
        ("", "causes", "Valve failure"),
        ("HAZOP-001", "", "Valve failure"),
        ("HAZOP-001", "causes", ""),
    ],
)
def test_rejects_blank_request_fields(
    scenario_id: str,
    predicate: str,
    known_endpoint: str,
) -> None:
    with pytest.raises(ValueError):
        RecoveryRequest(
            scenario_id=scenario_id,
            predicate=predicate,
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint=known_endpoint,
        )


def test_rejects_invalid_missing_endpoint() -> None:
    with pytest.raises(ValueError, match="MissingEndpoint"):
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint="invalid",  # type: ignore[arg-type]
            known_endpoint="Valve failure",
        )


def test_does_not_recover_relationship_from_another_scenario(
    repository: FakeGraphRepository,
) -> None:
    unrelated_relation = make_relation(
        object_="Tank overflow",
        evidence="Evidence from another scenario.",
        source_id="other-scenario-row",
    )

    repository.save_record(
        make_record(
            "HAZOP-OTHER",
            relations=(unrelated_relation,),
        )
    )

    result = RecoverMissingLink(ScenarioLocalRetriever(repository)).execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert result.scenario_id == "HAZOP-001"
    assert all(
        c.relation.source_id != "other-scenario-row"
        for c in result.candidates
    )
    assert {c.missing_value for c in result.candidates} == {
        "High tank level",
        "Tank overflow",
    }


def test_predicate_matching_is_case_insensitive(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="CAUSES",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert result.status == RecoveryStatus.CANDIDATES_FOUND


def test_endpoint_matching_is_case_insensitive(
    use_case: RecoverMissingLink,
) -> None:
    result = use_case.execute(
        RecoveryRequest(
            scenario_id="HAZOP-001",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="inlet valve fails open",
        )
    )

    assert result.status == RecoveryStatus.CANDIDATES_FOUND


def test_deduplicates_candidate_values_case_insensitively(
    repository: FakeGraphRepository,
) -> None:
    repository.save_record(
        make_record(
            "HAZOP-DUPLICATES",
            relations=(
                make_relation(
                    object_="High tank level",
                    source_id="source-001",
                ),
                make_relation(
                    object_="HIGH TANK LEVEL",
                    source_id="source-002",
                ),
            ),
        )
    )

    result = RecoverMissingLink(ScenarioLocalRetriever(repository)).execute(
        RecoveryRequest(
            scenario_id="HAZOP-DUPLICATES",
            predicate="causes",
            missing_endpoint=MissingEndpoint.OBJECT,
            known_endpoint="Inlet valve fails open",
        )
    )

    assert len(result.candidates) == 1
