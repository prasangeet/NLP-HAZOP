"""Tests for the missing-link reasoning orchestration use case."""

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.retrieval.scenario_local import ScenarioLocalRetriever
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.reason_about_missing_link import ReasonAboutMissingLink
from app.application.use_cases.recover_missing_link import RecoverMissingLink, RecoveryRequest
from app.domain.entities.hazop_record import CausalRelation, HazopRecord, RelationshipOrigin
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


class InMemoryRepository:
    def __init__(self) -> None:
        self.records: dict[str, HazopRecord] = {}

    def save_record(self, record: HazopRecord) -> None:
        self.records[record.scenario_id] = record

    def get_record(self, scenario_id: str) -> HazopRecord | None:
        return self.records.get(scenario_id)

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        limit: int = 20,
    ) -> list[HazopRecord]:
        records = list(self.records.values())
        if equipment is not None:
            records = [
                record for record in records
                if record.equipment.casefold() == equipment.casefold()
            ]
        if deviation is not None:
            records = [
                record for record in records
                if record.deviation.casefold() == deviation.casefold()
            ]
        return records[:limit]

    def save_relations(self, relations: list[CausalRelation]) -> None:
        return None

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        limit: int = 50,
    ) -> list[CausalRelation]:
        return []


def make_use_case(repository: InMemoryRepository) -> ReasonAboutMissingLink:
    return ReasonAboutMissingLink(
        recovery_use_case=RecoverMissingLink(ScenarioLocalRetriever(repository)),
        evidence_context_builder=BuildEvidenceContext(),
        reasoning_strategy=GraphReasoner(),
    )


def make_request() -> RecoveryRequest:
    return RecoveryRequest(
        scenario_id="HAZOP-001",
        predicate="causes",
        known_endpoint="Inlet valve fails open",
        missing_endpoint="object",
    )


def test_orchestrates_recovery_and_graph_reasoning() -> None:
    repository = InMemoryRepository()
    repository.save_record(
        HazopRecord(
            scenario_id="HAZOP-001",
            equipment="T-301",
            deviation="High level",
            causal_relations=(
                CausalRelation(
                    subject="Inlet valve fails open",
                    predicate="causes",
                    object="High tank level",
                    origin=RelationshipOrigin.SOURCE,
                    evidence="Source row 12 documents this relationship.",
                    source_id="source-row-12",
                    confidence=0.9,
                ),
            ),
        )
    )

    result = make_use_case(repository).execute(make_request())

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.object == "High tank level"


def test_returns_no_evidence_abstention_for_unknown_scenario() -> None:
    result = make_use_case(InMemoryRepository()).execute(make_request())

    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.NO_EVIDENCE
