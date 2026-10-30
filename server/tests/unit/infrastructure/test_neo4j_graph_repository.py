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


def test_dataset_import_uses_single_transaction_for_metadata_records_and_edges() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"data": []}
                    for _ in payload["statements"]
                ],
                "errors": [],
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

    assert len(requests) == 2
    schema_statements = requests[0]["statements"]
    import_statements = requests[1]["statements"]
    assert len(schema_statements) == 3
    assert len(import_statements) == 3
    assert "CREATE (d:HazopDataset" in import_statements[0]["statement"]
    assert "CREATE (s:HazopScenario" in import_statements[1]["statement"]
    assert "CAUSAL_EDGE" in import_statements[2]["statement"]
    assert import_statements[0]["parameters"]["source_sha256"] == "abc123"
    assert import_statements[1]["parameters"]["records"][0]["payload"]
    assert import_statements[2]["parameters"]["relations"][0]["origin"] == "synthetic"
    assert repository.storage_type == "neo4j"
    assert repository.is_persistent is True
    repository.close()


def test_get_record_deserializes_full_record_payload() -> None:
    record = sample_record()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "RETURN s.payload" in payload["statements"][0]["statement"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"data": [{"row": [json.dumps(record.to_dict())]}]}
                ],
                "errors": [],
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
