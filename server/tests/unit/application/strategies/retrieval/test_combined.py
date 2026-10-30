"""Tests for combined local and cross-scenario retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from app.application.strategies.retrieval.combined import CombinedRetriever
from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedEvidence,
    RetrievalScope,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)


@dataclass
class FakeRetriever:
    """A configurable retriever for isolated tests."""

    result: RetrievalResult

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return self.result


def make_relation(
    subject: str,
    predicate: str = "causes",
    object_: str = "high temperature",
    *,
    origin: RelationshipOrigin = RelationshipOrigin.SOURCE,
    evidence: str = "Supported by the source document.",
    source_id: str = "source-1",
) -> CausalRelation:
    """Create a causal relation for testing."""
    return CausalRelation(
        subject=subject,
        predicate=predicate,
        object=object_,
        origin=origin,
        evidence=evidence,
        confidence=0.9,
        source_id=source_id,
    )


def make_evidence(
    scenario_id: str,
    relation: CausalRelation,
    scope: RetrievalScope,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        scenario_id=scenario_id,
        relation=relation,
        scope=scope,
    )


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    return RetrievalRequest(
        scenario_id="T-301",
        predicate="causes",
        known_endpoint="pump failure",
        missing_endpoint="object",
        limit=3,
    )


def test_skips_cross_retrieval_when_local_limit_is_met(
    retrieval_request: RetrievalRequest,
) -> None:
    local_evidence = tuple(
        make_evidence(
            "T-301",
            make_relation(f"candidate {index}"),
            RetrievalScope.SCENARIO_LOCAL,
        )
        for index in range(retrieval_request.limit)
    )

    local = Mock()
    local.retrieve.return_value = RetrievalResult(
        scenario_id="T-301",
        scope=RetrievalScope.SCENARIO_LOCAL,
        evidence=local_evidence,
    )
    cross = Mock()

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert len(result.evidence) == 3
    assert all(
        item.scope == RetrievalScope.SCENARIO_LOCAL
        for item in result.evidence
    )
    cross.retrieve.assert_not_called()


def test_uses_cross_scenario_fallback_when_local_evidence_is_insufficient(
    retrieval_request: RetrievalRequest,
) -> None:
    local_item = make_evidence(
        "T-301",
        make_relation("local candidate"),
        RetrievalScope.SCENARIO_LOCAL,
    )
    cross_items = (
        make_evidence(
            "Scenario-2",
            make_relation("cross candidate 1"),
            RetrievalScope.CROSS_SCENARIO,
        ),
        make_evidence(
            "Scenario-3",
            make_relation("cross candidate 2"),
            RetrievalScope.CROSS_SCENARIO,
        ),
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = Mock()
    cross.retrieve.return_value = RetrievalResult(
        scenario_id="T-301",
        scope=RetrievalScope.CROSS_SCENARIO,
        evidence=cross_items,
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert len(result.evidence) == 3
    assert result.evidence[0] == local_item
    assert result.evidence[1:] == cross_items
    cross.retrieve.assert_called_once_with(retrieval_request)


def test_prioritizes_local_evidence(
    retrieval_request: RetrievalRequest,
) -> None:
    local_item = make_evidence(
        "T-301",
        make_relation("local candidate"),
        RetrievalScope.SCENARIO_LOCAL,
    )
    cross_item = make_evidence(
        "Scenario-2",
        make_relation("cross candidate"),
        RetrievalScope.CROSS_SCENARIO,
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=(cross_item,),
        )
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert result.evidence[0] == local_item
    assert result.evidence[1] == cross_item


def test_deduplicates_identical_relations_from_same_scenario(
    retrieval_request: RetrievalRequest,
) -> None:
    local_item = make_evidence(
        "T-301",
        make_relation("pump failure"),
        RetrievalScope.SCENARIO_LOCAL,
    )
    duplicate = make_evidence(
        "T-301",
        make_relation(
            "PUMP FAILURE",
            "CAUSES",
            "HIGH TEMPERATURE",
        ),
        RetrievalScope.CROSS_SCENARIO,
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=(duplicate,),
        )
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert len(result.evidence) == 1
    assert result.evidence[0] == local_item


def test_preserves_identical_relations_from_different_scenarios(
    retrieval_request: RetrievalRequest,
) -> None:
    relation = make_relation("pump failure")

    local_item = make_evidence(
        "T-301",
        relation,
        RetrievalScope.SCENARIO_LOCAL,
    )
    cross_item = make_evidence(
        "Scenario-2",
        relation,
        RetrievalScope.CROSS_SCENARIO,
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=(cross_item,),
        )
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert len(result.evidence) == 2
    assert result.evidence[0].scenario_id == "T-301"
    assert result.evidence[1].scenario_id == "Scenario-2"


def test_does_not_run_cross_retrieval_when_target_is_missing(
    retrieval_request: RetrievalRequest,
) -> None:
    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(),
            scenario_exists=False,
        )
    )
    cross = Mock()

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert result.scenario_exists is False
    assert result.evidence == ()
    assert result.scope == RetrievalScope.SCENARIO_LOCAL
    cross.retrieve.assert_not_called()


def test_result_does_not_exceed_requested_limit(
    retrieval_request: RetrievalRequest,
) -> None:
    local_item = make_evidence(
        "T-301",
        make_relation("local candidate"),
        RetrievalScope.SCENARIO_LOCAL,
    )
    cross_items = tuple(
        make_evidence(
            f"Scenario-{index}",
            make_relation(f"cross candidate {index}"),
            RetrievalScope.CROSS_SCENARIO,
        )
        for index in range(10)
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=cross_items,
        )
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert len(result.evidence) == retrieval_request.limit


def test_preserves_scenario_ids_scopes_and_source_provenance(
    retrieval_request: RetrievalRequest,
) -> None:
    local_item = make_evidence(
        "T-301",
        make_relation("local candidate", source_id="source-local"),
        RetrievalScope.SCENARIO_LOCAL,
    )
    cross_item = make_evidence(
        "DWSIM-ETHANOL-01",
        make_relation("cross candidate", source_id="source-dwsim"),
        RetrievalScope.CROSS_SCENARIO,
    )

    local = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.SCENARIO_LOCAL,
            evidence=(local_item,),
        )
    )
    cross = FakeRetriever(
        RetrievalResult(
            scenario_id="T-301",
            scope=RetrievalScope.CROSS_SCENARIO,
            evidence=(cross_item,),
        )
    )

    result = CombinedRetriever(local, cross).retrieve(retrieval_request)

    assert result.evidence[0].scenario_id == "T-301"
    assert result.evidence[0].scope == RetrievalScope.SCENARIO_LOCAL
    assert result.evidence[0].relation.source_id == "source-local"

    assert result.evidence[1].scenario_id == "DWSIM-ETHANOL-01"
    assert result.evidence[1].scope == RetrievalScope.CROSS_SCENARIO
    assert result.evidence[1].relation.source_id == "source-dwsim"
