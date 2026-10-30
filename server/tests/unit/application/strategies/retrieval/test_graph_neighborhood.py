"""Tests for leakage-aware graph-neighborhood retrieval."""

from app.application.strategies.retrieval.graph_neighborhood import (
    GraphNeighborhoodRetriever,
)
from app.application.strategies.retrieval.models import RetrievalRequest
from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
)
from app.infrastructure.graph.in_memory_graph_repository import (
    InMemoryGraphRepository,
)


def make_edge(origin: RelationshipOrigin) -> CausalRelation:
    return CausalRelation(
        subject="Valve failure",
        predicate="causes_deviation",
        object="High level",
        origin=origin,
        evidence="Test graph edge.",
        source_id=origin.value,
    )


def test_returns_only_other_scenarios_from_same_plant_unit() -> None:
    repository = InMemoryGraphRepository()
    repository.save_record(
        HazopRecord(
            scenario_id="HELD-OUT",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High level",
        )
    )
    repository.save_record(
        HazopRecord(
            scenario_id="TRAIN-SAME-UNIT",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High level",
            causal_relations=(make_edge(RelationshipOrigin.RECONSTRUCTED),),
        )
    )
    repository.save_record(
        HazopRecord(
            scenario_id="OTHER-UNIT",
            plant_unit="Reactor",
            equipment="R-101",
            deviation="High temperature",
            causal_relations=(make_edge(RelationshipOrigin.SOURCE),),
        )
    )
    repository.save_record(
        HazopRecord(
            scenario_id="MODEL-EDGE",
            plant_unit="Storage",
            equipment="T-301",
            deviation="High level",
            causal_relations=(make_edge(RelationshipOrigin.MODEL_INFERRED),),
        )
    )

    retriever = GraphNeighborhoodRetriever(
        ScenarioLocalRetriever(repository),
        repository,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            scenario_id="HELD-OUT",
            predicate="causes",
            known_endpoint="Valve failure",
            missing_endpoint="object",
        )
    )

    assert result.scenario_exists
    assert [item.scenario_id for item in result.graph_context] == [
        "TRAIN-SAME-UNIT"
    ]
    assert all(
        item.scenario_id != "HELD-OUT"
        for item in result.graph_context
    )
