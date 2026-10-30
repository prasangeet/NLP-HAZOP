"""Integration tests for persistent dataset, graph and recovery API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.infrastructure.graph.in_memory_graph_repository import (
    InMemoryGraphRepository,
)
from app.main import create_app


CSV = (
    "Scenario_ID,Plant/Unit,Node/Equipment,Deviation,"
    "Possible_Causes,Possible_Consequences\n"
    "S001,Feed section,P-101,No flow,Pump failure,"
    "Reduced feed supply\n"
)


class PersistentInMemoryRepository(InMemoryGraphRepository):
    """Test adapter with the persistence capability enabled."""

    storage_type = "neo4j"
    is_persistent = True


def build_client() -> tuple[TestClient, PersistentInMemoryRepository]:
    app = create_app()
    repository = PersistentInMemoryRepository()
    app.state.graph_repository = repository
    return TestClient(app), repository


def test_confirmed_import_and_dataset_query_flow() -> None:
    client, repository = build_client()
    with client:
        response = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
            data={"dataset_id": "feed-study", "name": "Feed system study"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "imported"
        assert payload["persisted"] is True
        assert payload["storage"] == "neo4j"
        assert payload["dataset"]["dataset_id"] == "feed-study"
        assert payload["dataset"]["record_count"] == 1

        listed = client.get("/api/v1/datasets")
        assert listed.status_code == 200
        assert listed.json()["count"] == 1

        scenarios = client.get("/api/v1/datasets/feed-study/scenarios")
        assert scenarios.status_code == 200
        assert scenarios.json()["count"] == 1
        record = scenarios.json()["scenarios"][0]
        assert record["scenario_id"] == "S001"
        assert record["raw_row"]["Scenario_ID"] == "S001"

        detail = client.get(
            "/api/v1/datasets/feed-study/scenarios/S001"
        )
        assert detail.status_code == 200

        graph = client.get("/api/v1/datasets/feed-study/graph")
        assert graph.status_code == 200
        assert graph.json()["node_count"] >= 2
        assert graph.json()["edge_count"] >= 2

        relations = client.get(
            "/api/v1/datasets/feed-study/relationships",
            params={"predicate": "causes_deviation"},
        )
        assert relations.status_code == 200
        assert relations.json()["count"] == 1

    assert repository.get_dataset("feed-study") is not None


def test_import_rejects_duplicate_dataset_id() -> None:
    client, _ = build_client()
    with client:
        first = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
            data={"dataset_id": "feed-study"},
        )
        second = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
            data={"dataset_id": "feed-study"},
        )

    assert first.status_code == 201
    assert second.status_code == 409


def test_import_rejects_duplicate_scenario_ids() -> None:
    client, _ = build_client()
    duplicate_csv = (
        "Scenario_ID,Deviation,Possible_Causes\n"
        "S001,No flow,Pump failure\n"
        "S001,High pressure,Valve blockage\n"
    )
    with client:
        response = client.post(
            "/api/v1/datasets/import",
            files={"file": ("duplicates.csv", duplicate_csv.encode(), "text/csv")},
        )

    assert response.status_code == 422
    assert "unique within a dataset" in response.json()["detail"]


def test_import_requires_a_persistent_repository() -> None:
    app = create_app()
    app.state.graph_repository = InMemoryGraphRepository()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
        )

    assert response.status_code == 503
    assert "requires persistent Neo4j storage" in response.json()["detail"]


def test_dataset_routes_return_not_found_for_unknown_dataset() -> None:
    client, _ = build_client()
    with client:
        response = client.get("/api/v1/datasets/missing/scenarios")

    assert response.status_code == 404


def test_recovery_endpoint_uses_existing_retrieval_and_reasoning() -> None:
    client, _ = build_client()
    with client:
        imported = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
            data={"dataset_id": "feed-study"},
        )
        assert imported.status_code == 201

        response = client.post(
            "/api/v1/datasets/feed-study/recovery",
            json={
                "scenario_id": "S001",
                "predicate": "causes_deviation",
                "known_endpoint": "Pump failure",
                "missing_endpoint": "object",
                "limit": 10,
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["recovery"]["status"] == "candidates_found"
    assert result["recovery"]["candidates"][0]["object"] == "No flow"
    assert result["reasoning"]["dataset_id"] == "feed-study"
    assert "not expert-validated causal facts" in result["notice"]


def test_recovery_rejects_unknown_scenario() -> None:
    client, _ = build_client()
    with client:
        imported = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV.encode(), "text/csv")},
            data={"dataset_id": "feed-study"},
        )
        assert imported.status_code == 201

        response = client.post(
            "/api/v1/datasets/feed-study/recovery",
            json={
                "scenario_id": "missing",
                "predicate": "causes_deviation",
                "known_endpoint": "Pump failure",
                "missing_endpoint": "object",
            },
        )

    assert response.status_code == 404
