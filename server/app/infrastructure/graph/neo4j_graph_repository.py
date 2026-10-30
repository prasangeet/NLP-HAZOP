"""Neo4j Query API adapter for the HAZOP graph repository.

Configure NEO4J_HTTP_URI with the Neo4j HTTP endpoint (for example,
http://127.0.0.1:7474), not the Bolt URI. Queries use Neo4j's current HTTP
Query API through the project's existing httpx dependency.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import ClassVar
from urllib.parse import quote

import httpx

from app.domain.entities.dataset import DatasetMetadata
from app.domain.ontology.schema import EntityType, infer_entity_types
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
    SourceReference,
    ValidationStatus,
)


class Neo4jRepositoryError(RuntimeError):
    """Raised when Neo4j rejects or cannot complete a repository operation."""


class Neo4jGraphRepository:
    """Persistent graph repository backed by Neo4j's HTTP Query API."""

    storage_type = "neo4j"
    is_persistent = True

    _SCHEMA_STATEMENTS: ClassVar[list[dict[str, object]]] = [
        {
            "statement": (
                "CREATE CONSTRAINT hazop_dataset_id IF NOT EXISTS "
                "FOR (d:HazopDataset) REQUIRE d.dataset_id IS UNIQUE"
            )
        },
        {
            "statement": (
                "CREATE CONSTRAINT hazop_scenario_identity IF NOT EXISTS "
                "FOR (s:HazopScenario) "
                "REQUIRE (s.dataset_id, s.scenario_id) IS UNIQUE"
            )
        },
        {
            "statement": (
                "CREATE CONSTRAINT hazop_concept_identity IF NOT EXISTS "
                "FOR (c:HazopConcept) "
                "REQUIRE (c.dataset_id, c.key) IS UNIQUE"
            )
        },
    ]

    def __init__(
        self,
        http_uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not http_uri.startswith(("http://", "https://")):
            raise ValueError(
                "NEO4J_HTTP_URI must use http:// or https://; Bolt URIs are not supported."
            )
        if not username.strip() or not password:
            raise ValueError("Neo4j username and password must be configured.")
        if not database.strip():
            raise ValueError("Neo4j database must not be empty.")

        self._endpoint = (
            f"{http_uri.rstrip('/')}/db/{quote(database.strip(), safe='')}/query/v2"
        )
        self._client = client or httpx.Client(
            auth=(username, password),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        self._schema_ready = False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def _execute(
        self,
        statements: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Execute parameterized queries through Neo4j Query API v2."""
        results: list[dict[str, object]] = []
        for item in statements:
            body: dict[str, object] = {
                "statement": item["statement"],
                "parameters": item.get("parameters", {}),
            }
            try:
                response = self._client.post(self._endpoint, json=body)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise Neo4jRepositoryError(
                    f"Neo4j Query API request failed: {exc}"
                ) from exc

            errors = payload.get("errors", [])
            if errors:
                details = "; ".join(
                    f"{error.get('code', 'Neo4jError')}: "
                    f"{error.get('message', 'Unknown error')}"
                    for error in errors
                )
                raise Neo4jRepositoryError(details)

            data = payload.get("data") or {}
            values = data.get("values", []) if isinstance(data, dict) else []
            results.append(
                {"data": [{"row": row} for row in values]}
            )
        return results

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self._execute(self._SCHEMA_STATEMENTS)
        self._schema_ready = True

    @staticmethod
    def _rows(result: dict[str, object]) -> list[list[object]]:
        data = result.get("data", [])
        return [item.get("row", []) for item in data]

    @staticmethod
    def _record_from_payload(payload: str | dict[str, object]) -> HazopRecord:
        data = json.loads(payload) if isinstance(payload, str) else payload
        source_data = data.get("source")
        source = (
            SourceReference(
                title=source_data["title"],
                document_id=source_data.get("document_id"),
                location=source_data.get("location"),
                provenance_note=source_data.get("provenance_note", ""),
            )
            if source_data
            else None
        )
        relations = tuple(
            CausalRelation(
                subject=item["subject"],
                predicate=item["predicate"],
                object=item["object"],
                origin=RelationshipOrigin(item["origin"]),
                evidence=item.get("evidence", ""),
                confidence=item.get("confidence"),
                source_id=item.get("source_id"),
                is_expert_verified=item.get("is_expert_verified", False),
                dataset_id=item.get("dataset_id", data.get("dataset_id", "default")),
                subject_type=EntityType(item.get("subject_type", infer_entity_types(item["predicate"], item["subject"], item["object"], deviation=data.get("deviation", ""))[0].value)),
                object_type=EntityType(item.get("object_type", infer_entity_types(item["predicate"], item["subject"], item["object"], deviation=data.get("deviation", ""))[1].value)),
                source_location=item.get("source_location"),
            )
            for item in data.get("causal_relations", [])
        )
        return HazopRecord(
            scenario_id=data["scenario_id"],
            dataset_id=data.get("dataset_id", "default"),
            plant_unit=data.get("plant_unit", ""),
            equipment=data.get("equipment", ""),
            design_intent=data.get("design_intent", ""),
            guideword=data.get("guideword", ""),
            parameter=data.get("parameter", ""),
            deviation=data.get("deviation", ""),
            causes=tuple(data.get("causes", [])),
            effects=tuple(data.get("effects", [])),
            consequences=tuple(data.get("consequences", [])),
            safeguards=tuple(data.get("safeguards", [])),
            recommendations=tuple(data.get("recommendations", [])),
            record_type=RecordType(data.get("record_type", "unspecified")),
            validation_status=ValidationStatus(
                data.get("validation_status", "unreviewed")
            ),
            source=source,
            causal_relations=relations,
            risk_ranking=data.get("risk_ranking"),
            remarks=data.get("remarks", ""),
            row_number=data.get("row_number"),
            raw_row=data.get("raw_row", {}),
        )

    @staticmethod
    def _dataset_from_properties(
        data: dict[str, object],
    ) -> DatasetMetadata:
        return DatasetMetadata(
            dataset_id=str(data["dataset_id"]),
            name=str(data["name"]),
            filename=str(data["filename"]),
            source_sha256=str(data["source_sha256"]),
            record_count=int(data["record_count"]),
            relationship_count=int(data["relationship_count"]),
            created_at=str(data["created_at"]),
        )

    @staticmethod
    def _relation_from_properties(
        data: dict[str, object],
        subject: str,
        object_value: str,
    ) -> CausalRelation:
        return CausalRelation(
            subject=subject,
            predicate=str(data["predicate"]),
            object=object_value,
            origin=RelationshipOrigin(str(data.get("origin", "unspecified"))),
            evidence=str(data.get("evidence", "")),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            source_id=(
                str(data["source_id"])
                if data.get("source_id") is not None
                else None
            ),
            is_expert_verified=bool(data.get("is_expert_verified", False)),
            dataset_id=str(data.get("dataset_id", "default")),
            subject_type=EntityType(str(data.get("subject_type", infer_entity_types(str(data["predicate"]), subject, object_value)[0].value))),
            object_type=EntityType(str(data.get("object_type", infer_entity_types(str(data["predicate"]), subject, object_value)[1].value))),
            source_location=(str(data["source_location"]) if data.get("source_location") is not None else None),
        )

    @staticmethod
    def _relation_rows(
        records: Sequence[HazopRecord],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for record in records:
            for index, relation in enumerate(record.causal_relations):
                inferred_subject_type, inferred_object_type = infer_entity_types(
                    relation.predicate,
                    relation.subject,
                    relation.object,
                    deviation=record.deviation,
                )
                subject_type = (
                    inferred_subject_type
                    if relation.subject_type == EntityType.CONCEPT
                    else relation.subject_type
                )
                object_type = (
                    inferred_object_type
                    if relation.object_type == EntityType.CONCEPT
                    else relation.object_type
                )
                rows.append(
                    {
                        "scenario_id": record.scenario_id,
                        "edge_id": (
                            f"{record.dataset_id}:{record.scenario_id}:{index}:"
                            f"{relation.subject.casefold()}:{relation.predicate}:"
                            f"{relation.object.casefold()}"
                        ),
                        "subject": relation.subject,
                        "subject_key": f"{subject_type.value}:{relation.subject.casefold()}",
                        "object": relation.object,
                        "object_key": f"{object_type.value}:{relation.object.casefold()}",
                        "predicate": relation.predicate,
                        "subject_type": subject_type.value,
                        "object_type": object_type.value,
                        "origin": relation.origin.value,
                        "evidence": relation.evidence,
                        "confidence": relation.confidence,
                        "source_id": relation.source_id,
                        "is_expert_verified": relation.is_expert_verified,
                    "source_location": relation.source_location,
                        "source_location": relation.source_location,
                    }
                )
        return rows

    def import_dataset(
        self,
        dataset: DatasetMetadata,
        records: Sequence[HazopRecord],
    ) -> None:
        """Atomically store dataset metadata, full records, and typed edges."""
        self._ensure_schema()
        record_rows = []
        for record in records:
            if record.dataset_id != dataset.dataset_id:
                raise ValueError("All records must belong to the imported dataset.")
            record_rows.append(
                {
                    "scenario_id": record.scenario_id,
                    "equipment": record.equipment,
                    "deviation": record.deviation,
                    "plant_unit": record.plant_unit,
                    "payload": json.dumps(record.to_dict(), ensure_ascii=False),
                }
            )

        if dataset.record_count != len(record_rows):
            raise ValueError("Dataset record_count does not match imported records.")
        if dataset.relationship_count != len(relation_rows := self._relation_rows(records)):
            raise ValueError(
                "Dataset relationship_count does not match imported relationships."
            )
        subject_label_sets = " ".join(
            f"FOREACH (_ IN CASE WHEN row.subject_type = '{entity.value}' THEN [1] ELSE [] END | SET source:{entity.value})"
            for entity in EntityType
            if entity != EntityType.CONCEPT
        )
        object_label_sets = " ".join(
            f"FOREACH (_ IN CASE WHEN row.object_type = '{entity.value}' THEN [1] ELSE [] END | SET target:{entity.value})"
            for entity in EntityType
            if entity != EntityType.CONCEPT
        )
        statement = (
            "CREATE (d:HazopDataset {"
            "dataset_id: $dataset_id, name: $name, filename: $filename, "
            "source_sha256: $source_sha256, record_count: $record_count, "
            "relationship_count: $relationship_count, created_at: $created_at"
            "}) "
            "WITH d "
            "UNWIND $records AS row "
            "CREATE (s:HazopScenario {"
            "dataset_id: $dataset_id, scenario_id: row.scenario_id, "
            "equipment: row.equipment, deviation: row.deviation, "
            "plant_unit: row.plant_unit, payload: row.payload"
            "}) "
            "CREATE (d)-[:CONTAINS_SCENARIO]->(s) "
            "WITH d, count(s) AS records_created "
            "CALL { "
            "WITH d "
            "UNWIND $relations AS row "
            "MERGE (source:HazopConcept {dataset_id: $dataset_id, key: row.subject_key}) "
            "SET source.name = row.subject "
            + subject_label_sets
            + " MERGE (target:HazopConcept {dataset_id: $dataset_id, key: row.object_key}) "
            "SET target.name = row.object "
            + object_label_sets
            + " MERGE (source)-[edge:CAUSAL_EDGE {edge_id: row.edge_id}]->(target) "
            "SET edge.dataset_id = $dataset_id, edge.scenario_id = row.scenario_id, "
            "edge.predicate = row.predicate, edge.subject_type = row.subject_type, "
            "edge.object_type = row.object_type, edge.origin = row.origin, "
            "edge.evidence = row.evidence, edge.confidence = row.confidence, "
            "edge.source_id = row.source_id, edge.is_expert_verified = row.is_expert_verified, "
                        "edge.source_location = row.source_location "
            "RETURN count(edge) AS relationships_created "
            "} "
            "RETURN d.dataset_id AS dataset_id, records_created, relationships_created"
        )
        parameters = {
            **dataset.to_dict(),
            "records": record_rows,
            "relations": relation_rows,
        }
        self._execute([{"statement": statement, "parameters": parameters}])

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        result = self._execute(
            [
                {
                    "statement": (
                        "MATCH (d:HazopDataset {dataset_id: $dataset_id}) "
                        "RETURN properties(d)"
                    ),
                    "parameters": {"dataset_id": dataset_id},
                }
            ]
        )[0]
        rows = self._rows(result)
        return self._dataset_from_properties(rows[0][0]) if rows else None

    def find_datasets(self, *, limit: int = 100) -> Sequence[DatasetMetadata]:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be greater than zero")
        result = self._execute(
            [
                {
                    "statement": (
                        "MATCH (d:HazopDataset) RETURN properties(d) "
                        "ORDER BY d.created_at DESC, d.dataset_id LIMIT $limit"
                    ),
                    "parameters": {"limit": limit},
                }
            ]
        )[0]
        return [
            self._dataset_from_properties(row[0])
            for row in self._rows(result)
        ]

    def save_record(self, record: HazopRecord) -> None:
        """Upsert a single record for benchmark and application use."""
        self._ensure_schema()
        self._execute(
            [
                {
                    "statement": (
                        "MERGE (s:HazopScenario {dataset_id: $dataset_id, "
                        "scenario_id: $scenario_id}) "
                        "SET s.equipment = $equipment, s.deviation = $deviation, "
                        "s.plant_unit = $plant_unit, s.payload = $payload"
                    ),
                    "parameters": {
                        "dataset_id": record.dataset_id,
                        "scenario_id": record.scenario_id,
                        "equipment": record.equipment,
                        "deviation": record.deviation,
                        "plant_unit": record.plant_unit,
                        "payload": json.dumps(record.to_dict(), ensure_ascii=False),
                    },
                },
            ]
        )
        self._save_relation_rows(
            record.dataset_id,
            self._relation_rows([record]),
        )

    def _save_relation_rows(
        self,
        dataset_id: str,
        rows: list[dict[str, object]],
    ) -> None:
        if not rows:
            return
        subject_label_sets = " ".join(
            f"FOREACH (_ IN CASE WHEN row.subject_type = '{entity.value}' THEN [1] ELSE [] END | SET source:{entity.value})"
            for entity in EntityType
            if entity != EntityType.CONCEPT
        )
        object_label_sets = " ".join(
            f"FOREACH (_ IN CASE WHEN row.object_type = '{entity.value}' THEN [1] ELSE [] END | SET target:{entity.value})"
            for entity in EntityType
            if entity != EntityType.CONCEPT
        )
        self._execute(
            [
                {
                    "statement": (
                        "UNWIND $relations AS row "
                        "MERGE (source:HazopConcept {dataset_id: $dataset_id, key: row.subject_key}) "
                        "SET source.name = row.subject "
                        + subject_label_sets
                        + " MERGE (target:HazopConcept {dataset_id: $dataset_id, key: row.object_key}) "
                        "SET target.name = row.object "
                        + object_label_sets
                        + " MERGE (source)-[edge:CAUSAL_EDGE {edge_id: row.edge_id}]->(target) "
                        "SET edge.dataset_id = $dataset_id, edge.scenario_id = row.scenario_id, "
                        "edge.predicate = row.predicate, edge.subject_type = row.subject_type, "
                        "edge.object_type = row.object_type, edge.origin = row.origin, "
                        "edge.evidence = row.evidence, edge.confidence = row.confidence, "
                        "edge.source_id = row.source_id, edge.is_expert_verified = row.is_expert_verified "
                        "RETURN count(edge)"
                    ),
                    "parameters": {"dataset_id": dataset_id, "relations": rows},
                }
            ]
        )

    def get_record(
        self, scenario_id: str, *, dataset_id: str = "default"
    ) -> HazopRecord | None:
        result = self._execute(
            [
                {
                    "statement": (
                        "MATCH (s:HazopScenario {dataset_id: $dataset_id, "
                        "scenario_id: $scenario_id}) RETURN s.payload"
                    ),
                    "parameters": {
                        "dataset_id": dataset_id,
                        "scenario_id": scenario_id,
                    },
                }
            ]
        )[0]
        rows = self._rows(result)
        return self._record_from_payload(rows[0][0]) if rows else None

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        dataset_id: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be greater than zero")
        result = self._execute(
            [
                {
                    "statement": (
                        "MATCH (s:HazopScenario) "
                        "WHERE ($dataset_id IS NULL OR s.dataset_id = $dataset_id) "
                        "AND ($equipment IS NULL OR toLower(s.equipment) = toLower($equipment)) "
                        "AND ($deviation IS NULL OR toLower(s.deviation) = toLower($deviation)) "
                        "RETURN s.payload ORDER BY s.dataset_id, s.scenario_id LIMIT $limit"
                    ),
                    "parameters": {
                        "dataset_id": dataset_id,
                        "equipment": equipment,
                        "deviation": deviation,
                        "limit": limit,
                    },
                }
            ]
        )[0]
        return [
            self._record_from_payload(row[0])
            for row in self._rows(result)
        ]

    def save_relations(self, relations: Sequence[CausalRelation]) -> None:
        """Upsert standalone relationships into the graph."""
        self._ensure_schema()
        rows = []
        for index, relation in enumerate(relations):
            inferred_subject_type, inferred_object_type = infer_entity_types(
                relation.predicate, relation.subject, relation.object
            )
            subject_type = (
                inferred_subject_type
                if relation.subject_type == EntityType.CONCEPT
                else relation.subject_type
            )
            object_type = (
                inferred_object_type
                if relation.object_type == EntityType.CONCEPT
                else relation.object_type
            )
            rows.append(
                {
                    "scenario_id": "unassigned",
                    "edge_id": (
                        f"{relation.dataset_id}:unassigned:{index}:"
                        f"{relation.subject.casefold()}:{relation.predicate}:"
                        f"{relation.object.casefold()}"
                    ),
                    "subject": relation.subject,
                    "subject_key": f"{subject_type.value}:{relation.subject.casefold()}",
                    "object": relation.object,
                    "object_key": f"{object_type.value}:{relation.object.casefold()}",
                    "predicate": relation.predicate,
                    "subject_type": subject_type.value,
                    "object_type": object_type.value,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "confidence": relation.confidence,
                    "source_id": relation.source_id,
                    "is_expert_verified": relation.is_expert_verified,
                }
            )
        grouped: dict[str, list[dict[str, object]]] = {}
        for row, relation in zip(rows, relations, strict=True):
            grouped.setdefault(relation.dataset_id, []).append(row)
        for dataset_id, dataset_rows in grouped.items():
            self._save_relation_rows(dataset_id, dataset_rows)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        dataset_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be greater than zero")
        result = self._execute(
            [
                {
                    "statement": (
                        "MATCH (source:HazopConcept)-[edge:CAUSAL_EDGE]->"
                        "(target:HazopConcept) "
                        "WHERE ($dataset_id IS NULL OR edge.dataset_id = $dataset_id) "
                        "AND ($subject IS NULL OR toLower(source.name) = toLower($subject)) "
                        "AND ($predicate IS NULL OR toLower(edge.predicate) = toLower($predicate)) "
                        "AND ($object IS NULL OR toLower(target.name) = toLower($object)) "
                        "RETURN properties(edge), source.name, target.name "
                        "ORDER BY edge.dataset_id, edge.scenario_id, edge.predicate "
                        "LIMIT $limit"
                    ),
                    "parameters": {
                        "dataset_id": dataset_id,
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "limit": limit,
                    },
                }
            ]
        )[0]
        return [
            self._relation_from_properties(row[0], str(row[1]), str(row[2]))
            for row in self._rows(result)
        ]
