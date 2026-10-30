"""Unit tests for the Neo4j transactional HTTP graph repository."""

from __future__ import annotations

import json

import httpx

from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RelationshipOrigin,
)
from app.infrastructure.graph.neo4j_graph_repository import (
    Neo4jGraphRepository,
)


def sample_record() -> HazopRecord:
    relation = CausalRelation(
        subject="Pump failure",
        predicate="causes_deviation",
        object="No flow",
        origin=RelationshipOrigin.SYNTHETIC,
        evidence="Co-listed in a synthetic worksheet row.",
        source_id="feed-study:S001:2:cause:0",
        dataset_id="feed-study",
    )
    return HazopRecord(
        scenario_id="S001",
        dataset_id="feed-study",
        equipment="P-101",
        deviation="No flow",
        causes=("Pump failure",),
        causal_relations=(relation,),
        raw_row={"Scenario_ID": "S001"},
        row_number=2,
    )


def sample_dataset() -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="feed-study",
        name="Feed study",
        filename="feed.csv",
        source_sha256="abc123",
        record_count=1,
        relationship_count=1,
        created_at="2026-10-31T00:00:00+00:00",
    )


def test_dataset_import_is_one_atomic_query_api_transaction() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append({"path": request.url.path, "payload": payload})
        return httpx.Response(
            202,
            json={
                "data": {"fields": [], "values": []},
                "errors": [],
                "bookmarks": [],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    repository = Neo4jGraphRepository(
        "http://neo4j.test:7474",
        "neo4j",
        "secret",
        client=client,
    )
    repository.import_dataset(sample_dataset(), [sample_record()])

    # Three schema constraints plus one atomic dataset write.
    assert len(requests) == 4
    assert all(
        request["path"] == "/db/neo4j/query/v2"
        for request in requests
    )
    schema_statements = [
        request["payload"]["statement"]
        for request in requests[:3]
    ]
    assert len(schema_statements) == 3
    assert all("CREATE CONSTRAINT" in statement for statement in schema_statements)

    import_request = requests[3]["payload"]
    import_statement = import_request["statement"]
    parameters = import_request["parameters"]
    assert "CREATE (d:HazopDataset" in import_statement
    assert "CREATE (s:HazopScenario" in import_statement
    assert "CAUSAL_EDGE" in import_statement
    assert "CALL {" in import_statement
    assert parameters["source_sha256"] == "abc123"
    assert parameters["records"][0]["payload"]
    assert parameters["relations"][0]["origin"] == "synthetic"
    assert parameters["relations"][0]["subject_type"] == "Cause"
    assert parameters["relations"][0]["object_type"] == "Deviation"
    assert parameters["relations"][0]["subject_key"] == "Cause:pump failure"
    assert parameters["relations"][0]["object_key"] == "Deviation:no flow"
    assert repository.storage_type == "neo4j"
    assert repository.is_persistent is True
    repository.close()


def test_get_record_deserializes_full_record_payload() -> None:
    record = sample_record()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "RETURN s.payload" in payload["statement"]
        return httpx.Response(
            202,
            json={
                "data": {
                    "fields": ["s.payload"],
                    "values": [[json.dumps(record.to_dict())]],
                },
                "errors": [],
                "bookmarks": [],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    repository = Neo4jGraphRepository(
        "http://neo4j.test:7474",
        "neo4j",
        "secret",
        client=client,
    )
    restored = repository.get_record("S001", dataset_id="feed-study")

    assert restored == record
    assert restored is not None
    assert restored.causal_relations[0].origin == RelationshipOrigin.SYNTHETIC
    assert restored.raw_row == {"Scenario_ID": "S001"}
    repository.close()


def test_neo4j_repository_rejects_bolt_uri() -> None:
    try:
        Neo4jGraphRepository("bolt://localhost:7687", "neo4j", "secret")
    except ValueError as exc:
        assert "http://" in str(exc)
    else:
        raise AssertionError("Expected invalid URI to fail")
