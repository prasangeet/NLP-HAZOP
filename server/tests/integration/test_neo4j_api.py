"""Live Neo4j Query API integration test.

This test is run only in the CI job that provisions a disposable Neo4j service.
It exercises the real Cypher, transaction semantics, serialization, and API wiring.
"""

from __future__ import annotations

from os import environ
from time import sleep
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from pytest import mark


NEO4J_CONFIGURED = all(
    environ.get(key)
    for key in ("NEO4J_HTTP_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
)
pytestmark = mark.skipif(
    not NEO4J_CONFIGURED,
    reason="requires NEO4J_HTTP_URI, NEO4J_USERNAME, and NEO4J_PASSWORD",
)


def _wait_for_neo4j() -> None:
    """Wait until the authenticated Query API is ready in the service container."""
    endpoint = (
        environ["NEO4J_HTTP_URI"].rstrip("/")
        + "/db/"
        + environ.get("NEO4J_DATABASE", "neo4j")
        + "/query/v2"
    )
    auth = (environ["NEO4J_USERNAME"], environ["NEO4J_PASSWORD"])
    with httpx.Client(auth=auth, timeout=3.0) as client:
        for _ in range(60):
            try:
                response = client.post(endpoint, json={"statement": "RETURN 1"})
                if response.status_code == 202 and not response.json().get("errors"):
                    return
            except (httpx.HTTPError, ValueError):
                pass
            sleep(1)
    raise RuntimeError("Neo4j Query API did not become ready within 60 seconds.")


def _create_app():
    """Import the app factory only when the live Neo4j test is enabled."""
    from app.main import create_app

    return create_app()


CSV_CONTENT = (
    "Scenario_ID,Plant/Unit,Node/Equipment,Deviation,"
    "Possible_Causes,Possible_Consequences\n"
    "S001,Feed section,P-101,No flow,Pump failure,"
    "Reduced feed supply\n"
)


def test_import_persists_across_app_restart_and_supports_graph_recovery() -> None:
    _wait_for_neo4j()
    dataset_id = f"ci-{uuid4().hex}"
    first_app = _create_app()

    with TestClient(first_app) as client:
        repository = first_app.state.graph_repository
        assert repository is not None
        assert repository.storage_type == "neo4j"
        assert repository.is_persistent is True

        preview = client.post(
            "/api/v1/datasets/preview",
            files={"file": ("feed.csv", CSV_CONTENT.encode(), "text/csv")},
        )
        assert preview.status_code == 200
        assert preview.json()["persisted"] is False

        imported = client.post(
            "/api/v1/datasets/import",
            files={"file": ("feed.csv", CSV_CONTENT.encode(), "text/csv")},
            data={"dataset_id": dataset_id, "name": "CI feed study"},
        )
        assert imported.status_code == 201, imported.text
        assert imported.json()["persisted"] is True
        assert imported.json()["dataset"]["record_count"] == 1

    # A new app creates a new repository/client. Reads must come from Neo4j,
    # not from the first app's process memory.
    second_app = _create_app()
    with TestClient(second_app) as client:
        metadata = client.get(f"/api/v1/datasets/{dataset_id}")
        assert metadata.status_code == 200
        assert metadata.json()["dataset"]["name"] == "CI feed study"

        scenarios = client.get(f"/api/v1/datasets/{dataset_id}/scenarios")
        assert scenarios.status_code == 200
        assert scenarios.json()["count"] == 1
        scenario = scenarios.json()["scenarios"][0]
        assert scenario["scenario_id"] == "S001"
        assert scenario["raw_row"]["Scenario_ID"] == "S001"

        relations = client.get(
            f"/api/v1/datasets/{dataset_id}/relationships",
            params={"predicate": "causes_deviation"},
        )
        assert relations.status_code == 200
        assert relations.json()["count"] == 1

        graph = client.get(f"/api/v1/datasets/{dataset_id}/graph")
        assert graph.status_code == 200
        assert graph.json()["node_count"] >= 2
        assert graph.json()["edge_count"] >= 2

        recovery = client.post(
            f"/api/v1/datasets/{dataset_id}/recovery",
            json={
                "scenario_id": "S001",
                "predicate": "causes_deviation",
                "known_endpoint": "Pump failure",
                "missing_endpoint": "object",
                "limit": 10,
            },
        )
        assert recovery.status_code == 200, recovery.text
        payload = recovery.json()
        assert payload["recovery"]["status"] == "candidates_found"
        assert payload["recovery"]["candidates"][0]["object"] == "No flow"
        assert "not expert-validated causal facts" in payload["notice"]
