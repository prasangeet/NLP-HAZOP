"""Neo4j Query API adapter for the HAZOP graph repository.

Configure NEO4J_HTTP_URI with the Neo4j HTTP endpoint, for example:
http://127.0.0.1:7474

This adapter uses Neo4j Query API v2 through httpx.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import ClassVar
from urllib.parse import quote

import httpx

from app.domain.entities.dataset import DatasetMetadata
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
    SourceReference,
    ValidationStatus,
)
from app.domain.ontology.schema import EntityType, infer_entity_types


class Neo4jRepositoryError(RuntimeError):
    """Raised when Neo4j rejects or cannot complete a repository operation."""


class Neo4jGraphRepository:
    """Persistent graph repository backed by Neo4j's HTTP Query API."""

    storage_type = "neo4j"
    is_persistent = True

    _SCHEMA_STATEMENTS: ClassVar[tuple[str, ...]] = (
        (
            "CREATE CONSTRAINT hazop_dataset_id IF NOT EXISTS "
            "FOR (d:HazopDataset) REQUIRE d.dataset_id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT hazop_scenario_identity IF NOT EXISTS "
            "FOR (s:HazopScenario) "
            "REQUIRE (s.dataset_id, s.scenario_id) IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT hazop_concept_identity IF NOT EXISTS "
            "FOR (c:HazopConcept) "
            "REQUIRE (c.dataset_id, c.key) IS UNIQUE"
        ),
    )

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
                "NEO4J_HTTP_URI must use http:// or https://; "
                "Bolt URIs are not supported."
            )
        if not username.strip() or not password:
            raise ValueError("Neo4j username and password must be configured.")
        if not database.strip():
            raise ValueError("Neo4j database must not be empty.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._endpoint = (
            f"{http_uri.rstrip('/')}/db/"
            f"{quote(database.strip(), safe='')}/query/v2"
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

    # ------------------------------------------------------------------
    # Query API and response validation
    # ------------------------------------------------------------------

    @staticmethod
    def _as_mapping(value: object, *, context: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j response: expected an object for {context}."
            )
        if not all(isinstance(key, str) for key in value):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j response: non-string key in {context}."
            )
        return value

    @staticmethod
    def _required_string(
        data: Mapping[str, object],
        key: str,
        *,
        context: str,
    ) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j response: {context}.{key} must be a string."
            )
        return value

    @staticmethod
    def _optional_string(
        data: Mapping[str, object],
        key: str,
        *,
        default: str | None = None,
    ) -> str | None:
        value = data.get(key, default)
        if value is None:
            return None
        if not isinstance(value, str):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j response: {key} must be a string or null."
            )
        return value

    @staticmethod
    def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j payload: {field_name} must be a list of strings."
            )
        return tuple(value)

    @staticmethod
    def _integer(value: object, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise Neo4jRepositoryError(
                f"Invalid Neo4j response: {field_name} must be an integer."
            )
        return value

    def _execute(
        self,
        statements: Sequence[tuple[str, Mapping[str, object]]],
    ) -> list[list[list[object]]]:
        """Execute parameterized statements and return their result rows."""
        results: list[list[list[object]]] = []

        for statement, parameters in statements:
            body = {
                "statement": statement,
                "parameters": dict(parameters),
            }

            try:
                response = self._client.post(self._endpoint, json=body)
                response.raise_for_status()
                payload_object = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise Neo4jRepositoryError(
                    f"Neo4j Query API request failed: {exc}"
                ) from exc

            payload = self._as_mapping(
                payload_object,
                context="response",
            )

            errors = payload.get("errors", [])
            if not isinstance(errors, list):
                raise Neo4jRepositoryError(
                    "Invalid Neo4j response: errors must be a list."
                )

            if errors:
                messages: list[str] = []
                for error_object in errors:
                    error = self._as_mapping(error_object, context="error")
                    code = error.get("code", "Neo4jError")
                    message = error.get("message", "Unknown Neo4j error")
                    messages.append(f"{code}: {message}")
                raise Neo4jRepositoryError("; ".join(messages))

            data = self._as_mapping(
                payload.get("data", {}),
                context="data",
            )
            values = data.get("values", [])

            if not isinstance(values, list):
                raise Neo4jRepositoryError(
                    "Invalid Neo4j response: data.values must be a list."
                )

            rows: list[list[object]] = []
            for row_object in values:
                if not isinstance(row_object, list):
                    raise Neo4jRepositoryError(
                        "Invalid Neo4j response: each result row must be a list."
                    )
                rows.append(row_object)

            results.append(rows)

        return results

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        self._execute(
            [(statement, {}) for statement in self._SCHEMA_STATEMENTS]
        )
        self._schema_ready = True

    # ------------------------------------------------------------------
    # Serialization and domain conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _json_payload(record: HazopRecord) -> str:
        try:
            return json.dumps(record.to_dict(), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record '{record.scenario_id}' cannot be serialized."
            ) from exc

    @staticmethod
    def _decode_record_payload(payload: object) -> Mapping[str, object]:
        if isinstance(payload, str):
            try:
                decoded: object = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise Neo4jRepositoryError(
                    "Stored HAZOP record contains invalid JSON."
                ) from exc
        else:
            decoded = payload

        return Neo4jGraphRepository._as_mapping(
            decoded,
            context="record payload",
        )

    @staticmethod
    def _record_from_payload(payload: object) -> HazopRecord:
        data = Neo4jGraphRepository._decode_record_payload(payload)

        scenario_id = Neo4jGraphRepository._required_string(
            data, "scenario_id", context="record"
        )
        dataset_id = Neo4jGraphRepository._optional_string(
            data, "dataset_id", default="default"
        )
        if dataset_id is None:
            dataset_id = "default"

        deviation = Neo4jGraphRepository._optional_string(
            data, "deviation", default=""
        ) or ""

        source_data = data.get("source")
        source: SourceReference | None = None

        if source_data is not None:
            source_mapping = Neo4jGraphRepository._as_mapping(
                source_data,
                context="record.source",
            )
            source_title = Neo4jGraphRepository._required_string(
                source_mapping,
                "title",
                context="record.source",
            )
            source = SourceReference(
                title=source_title,
                document_id=Neo4jGraphRepository._optional_string(
                    source_mapping, "document_id"
                ),
                location=Neo4jGraphRepository._optional_string(
                    source_mapping, "location"
                ),
                provenance_note=(
                    Neo4jGraphRepository._optional_string(
                        source_mapping,
                        "provenance_note",
                        default="",
                    )
                    or ""
                ),
            )

        raw_relations = data.get("causal_relations", [])
        if not isinstance(raw_relations, list):
            raise Neo4jRepositoryError(
                "Invalid record payload: causal_relations must be a list."
            )

        relations: list[CausalRelation] = []

        for relation_object in raw_relations:
            item = Neo4jGraphRepository._as_mapping(
                relation_object,
                context="causal relation",
            )

            subject = Neo4jGraphRepository._required_string(
                item, "subject", context="causal relation"
            )
            predicate = Neo4jGraphRepository._required_string(
                item, "predicate", context="causal relation"
            )
            object_value = Neo4jGraphRepository._required_string(
                item, "object", context="causal relation"
            )

            inferred_subject_type, inferred_object_type = infer_entity_types(
                predicate,
                subject,
                object_value,
                deviation=deviation,
            )

            subject_type_value = item.get(
                "subject_type", inferred_subject_type.value
            )
            object_type_value = item.get(
                "object_type", inferred_object_type.value
            )

            if not isinstance(subject_type_value, str):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: subject_type must be a string."
                )
            if not isinstance(object_type_value, str):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: object_type must be a string."
                )

            confidence_value = item.get("confidence")
            if confidence_value is not None and (
                isinstance(confidence_value, bool)
                or not isinstance(confidence_value, (int, float))
            ):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: confidence must be numeric."
                )

            verified_value = item.get("is_expert_verified", False)
            if not isinstance(verified_value, bool):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: is_expert_verified "
                    "must be a boolean."
                )

            relation_dataset_id = item.get("dataset_id", dataset_id)
            if not isinstance(relation_dataset_id, str):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: dataset_id must be a string."
                )

            evidence = item.get("evidence", "")
            if not isinstance(evidence, str):
                raise Neo4jRepositoryError(
                    "Invalid relation payload: evidence must be a string."
                )

            relations.append(
                CausalRelation(
                    subject=subject,
                    predicate=predicate,
                    object=object_value,
                    origin=RelationshipOrigin(
                        str(item.get("origin", "unspecified"))
                    ),
                    evidence=evidence,
                    confidence=(
                        float(confidence_value)
                        if confidence_value is not None
                        else None
                    ),
                    source_id=Neo4jGraphRepository._optional_string(
                        item, "source_id"
                    ),
                    is_expert_verified=verified_value,
                    dataset_id=relation_dataset_id,
                    subject_type=EntityType(subject_type_value),
                    object_type=EntityType(object_type_value),
                    source_location=Neo4jGraphRepository._optional_string(
                        item, "source_location"
                    ),
                )
            )

        def optional_text(field_name: str, default: str = "") -> str:
            value = data.get(field_name, default)
            if not isinstance(value, str):
                raise Neo4jRepositoryError(
                    f"Invalid record payload: {field_name} must be a string."
                )
            return value

        risk_ranking = data.get("risk_ranking")
        if risk_ranking is not None and not isinstance(risk_ranking, str):
            raise Neo4jRepositoryError(
                "Invalid record payload: risk_ranking must be a string or null."
            )

        row_number = data.get("row_number")
        if row_number is not None and (
            isinstance(row_number, bool) or not isinstance(row_number, int)
        ):
            raise Neo4jRepositoryError(
                "Invalid record payload: row_number must be an integer or null."
            )

        raw_row_object = data.get("raw_row", {})
        raw_row_mapping = Neo4jGraphRepository._as_mapping(
            raw_row_object,
            context="record.raw_row",
        )

        return HazopRecord(
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            plant_unit=optional_text("plant_unit"),
            equipment=optional_text("equipment"),
            design_intent=optional_text("design_intent"),
            guideword=optional_text("guideword"),
            parameter=optional_text("parameter"),
            deviation=deviation,
            causes=Neo4jGraphRepository._string_tuple(
                data.get("causes", []), field_name="causes"
            ),
            effects=Neo4jGraphRepository._string_tuple(
                data.get("effects", []), field_name="effects"
            ),
            consequences=Neo4jGraphRepository._string_tuple(
                data.get("consequences", []), field_name="consequences"
            ),
            safeguards=Neo4jGraphRepository._string_tuple(
                data.get("safeguards", []), field_name="safeguards"
            ),
            recommendations=Neo4jGraphRepository._string_tuple(
                data.get("recommendations", []), field_name="recommendations"
            ),
            record_type=RecordType(
                optional_text("record_type", "unspecified")
            ),
            validation_status=ValidationStatus(
                optional_text("validation_status", "unreviewed")
            ),
            source=source,
            causal_relations=tuple(relations),
            risk_ranking=risk_ranking,
            remarks=optional_text("remarks"),
            row_number=row_number,
            raw_row=dict(raw_row_mapping),
        )

    @staticmethod
    def _dataset_from_properties(data_object: object) -> DatasetMetadata:
        data = Neo4jGraphRepository._as_mapping(
            data_object,
            context="dataset properties",
        )

        return DatasetMetadata(
            dataset_id=Neo4jGraphRepository._required_string(
                data, "dataset_id", context="dataset"
            ),
            name=Neo4jGraphRepository._required_string(
                data, "name", context="dataset"
            ),
            filename=Neo4jGraphRepository._required_string(
                data, "filename", context="dataset"
            ),
            source_sha256=Neo4jGraphRepository._required_string(
                data, "source_sha256", context="dataset"
            ),
            record_count=Neo4jGraphRepository._integer(
                data.get("record_count"),
                field_name="dataset.record_count",
            ),
            relationship_count=Neo4jGraphRepository._integer(
                data.get("relationship_count"),
                field_name="dataset.relationship_count",
            ),
            created_at=Neo4jGraphRepository._required_string(
                data, "created_at", context="dataset"
            ),
        )

    @staticmethod
    def _relation_from_properties(
        data_object: object,
        subject: object,
        object_value: object,
    ) -> CausalRelation:
        data = Neo4jGraphRepository._as_mapping(
            data_object,
            context="relationship properties",
        )

        if not isinstance(subject, str) or not isinstance(object_value, str):
            raise Neo4jRepositoryError(
                "Invalid Neo4j response: relationship endpoints must be strings."
            )

        predicate = Neo4jGraphRepository._required_string(
            data, "predicate", context="relationship"
        )

        inferred_subject_type, inferred_object_type = infer_entity_types(
            predicate, subject, object_value
        )

        confidence_value = data.get("confidence")
        if confidence_value is not None and (
            isinstance(confidence_value, bool)
            or not isinstance(confidence_value, (int, float))
        ):
            raise Neo4jRepositoryError(
                "Invalid relationship: confidence must be numeric or null."
            )

        verified_value = data.get("is_expert_verified", False)
        if not isinstance(verified_value, bool):
            raise Neo4jRepositoryError(
                "Invalid relationship: is_expert_verified must be boolean."
            )

        evidence = data.get("evidence", "")
        if not isinstance(evidence, str):
            raise Neo4jRepositoryError(
                "Invalid relationship: evidence must be a string."
            )

        return CausalRelation(
            subject=subject,
            predicate=predicate,
            object=object_value,
            origin=RelationshipOrigin(
                str(data.get("origin", "unspecified"))
            ),
            evidence=evidence,
            confidence=(
                float(confidence_value)
                if confidence_value is not None
                else None
            ),
            source_id=Neo4jGraphRepository._optional_string(
                data, "source_id"
            ),
            is_expert_verified=verified_value,
            dataset_id=(
                Neo4jGraphRepository._optional_string(
                    data, "dataset_id", default="default"
                )
                or "default"
            ),
            subject_type=EntityType(
                str(data.get("subject_type", inferred_subject_type.value))
            ),
            object_type=EntityType(
                str(data.get("object_type", inferred_object_type.value))
            ),
            source_location=Neo4jGraphRepository._optional_string(
                data, "source_location"
            ),
        )

    # ------------------------------------------------------------------
    # Relationship identity and graph row preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_hash(value: Mapping[str, object]) -> str:
        serialized = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _typed_relation(
        relation: CausalRelation,
        *,
        deviation: str = "",
    ) -> tuple[EntityType, EntityType]:
        inferred_subject_type, inferred_object_type = infer_entity_types(
            relation.predicate,
            relation.subject,
            relation.object,
            deviation=deviation,
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
        return subject_type, object_type

    @classmethod
    def _relation_rows(
        cls,
        records: Sequence[HazopRecord],
    ) -> list[dict[str, object]]:
        """Prepare stable, individually identifiable relationship rows."""
        rows: list[dict[str, object]] = []

        for record in records:
            occurrences: Counter[str] = Counter()

            for relation in record.causal_relations:
                subject_type, object_type = cls._typed_relation(
                    relation,
                    deviation=record.deviation,
                )

                identity: dict[str, object] = {
                    "dataset_id": record.dataset_id,
                    "scenario_id": record.scenario_id,
                    "source_id": relation.source_id,
                    "subject": relation.subject.casefold(),
                    "predicate": relation.predicate.casefold(),
                    "object": relation.object.casefold(),
                    "subject_type": subject_type.value,
                    "object_type": object_type.value,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "source_location": relation.source_location,
                }

                signature = cls._stable_hash(identity)
                occurrence = occurrences[signature]
                occurrences[signature] += 1

                edge_id = cls._stable_hash(
                    {
                        "identity": identity,
                        "occurrence": occurrence,
                    }
                )

                rows.append(
                    cls._make_relation_row(
                        relation,
                        scenario_id=record.scenario_id,
                        edge_id=edge_id,
                        subject_type=subject_type,
                        object_type=object_type,
                    )
                )

        return rows

    @classmethod
    def _standalone_relation_rows(
        cls,
        relations: Sequence[CausalRelation],
    ) -> list[dict[str, object]]:
        """Prepare stable relationship rows without a scenario record."""
        rows: list[dict[str, object]] = []
        occurrences: Counter[str] = Counter()

        for relation in relations:
            subject_type, object_type = cls._typed_relation(relation)

            identity: dict[str, object] = {
                "dataset_id": relation.dataset_id,
                "source_id": relation.source_id,
                "subject": relation.subject.casefold(),
                "predicate": relation.predicate.casefold(),
                "object": relation.object.casefold(),
                "subject_type": subject_type.value,
                "object_type": object_type.value,
                "origin": relation.origin.value,
                "evidence": relation.evidence,
                "source_location": relation.source_location,
            }

            signature = cls._stable_hash(identity)
            occurrence = occurrences[signature]
            occurrences[signature] += 1

            edge_id = cls._stable_hash(
                {
                    "identity": identity,
                    "occurrence": occurrence,
                }
            )

            rows.append(
                cls._make_relation_row(
                    relation,
                    scenario_id="unassigned",
                    edge_id=edge_id,
                    subject_type=subject_type,
                    object_type=object_type,
                )
            )

        return rows

    @staticmethod
    def _make_relation_row(
        relation: CausalRelation,
        *,
        scenario_id: str,
        edge_id: str,
        subject_type: EntityType,
        object_type: EntityType,
    ) -> dict[str, object]:
        return {
            "scenario_id": scenario_id,
            "edge_id": edge_id,
            "subject": relation.subject,
            "subject_key": (
                f"{subject_type.value}:{relation.subject.casefold()}"
            ),
            "object": relation.object,
            "object_key": (
                f"{object_type.value}:{relation.object.casefold()}"
            ),
            "predicate": relation.predicate,
            "subject_type": subject_type.value,
            "object_type": object_type.value,
            "origin": relation.origin.value,
            "evidence": relation.evidence,
            "confidence": relation.confidence,
            "source_id": relation.source_id,
            "is_expert_verified": relation.is_expert_verified,
            "source_location": relation.source_location,
        }

    @staticmethod
    def _label_statements() -> tuple[str, str, str, str]:
        """Build label operations from the controlled EntityType enum."""
        labels = [
            entity.value
            for entity in EntityType
            if entity != EntityType.CONCEPT
        ]

        if not labels:
            return "", "", "", ""

        remove_source = "REMOVE source:" + ":".join(labels) + " "
        remove_target = "REMOVE target:" + ":".join(labels) + " "

        set_source = " ".join(
            (
                "FOREACH (_ IN CASE "
                f"WHEN row.subject_type = '{label}' "
                f"THEN [1] ELSE [] END | SET source:{label})"
            )
            for label in labels
        )
        set_target = " ".join(
            (
                "FOREACH (_ IN CASE "
                f"WHEN row.object_type = '{label}' "
                f"THEN [1] ELSE [] END | SET target:{label})"
            )
            for label in labels
        )
        return remove_source, remove_target, set_source, set_target

    def _upsert_relation_rows(
        self,
        dataset_id: str,
        rows: Sequence[dict[str, object]],
    ) -> None:
        if not rows:
            return

        remove_source, remove_target, set_source, set_target = (
            self._label_statements()
        )

        statement = (
            "UNWIND $relations AS row "
            "MERGE (source:HazopConcept "
            "{dataset_id: $dataset_id, key: row.subject_key}) "
            "SET source.name = row.subject "
            + remove_source
            + set_source
            + " MERGE (target:HazopConcept "
            "{dataset_id: $dataset_id, key: row.object_key}) "
            "SET target.name = row.object "
            + remove_target
            + set_target
            + " MERGE (source)-[edge:CAUSAL_EDGE "
            "{edge_id: row.edge_id}]->(target) "
            "SET edge.dataset_id = $dataset_id, "
            "edge.scenario_id = row.scenario_id, "
            "edge.predicate = row.predicate, "
            "edge.subject_type = row.subject_type, "
            "edge.object_type = row.object_type, "
            "edge.origin = row.origin, "
            "edge.evidence = row.evidence, "
            "edge.confidence = row.confidence, "
            "edge.source_id = row.source_id, "
            "edge.is_expert_verified = row.is_expert_verified, "
            "edge.source_location = row.source_location "
            "RETURN count(edge)"
        )

        self._execute(
            [
                (
                    statement,
                    {
                        "dataset_id": dataset_id,
                        "relations": list(rows),
                    },
                )
            ]
        )

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------

    def import_dataset(
        self,
        dataset: DatasetMetadata,
        records: Sequence[HazopRecord],
    ) -> None:
        """Atomically create dataset metadata, records and graph edges."""
        self._ensure_schema()

        record_rows: list[dict[str, object]] = []
        seen_scenarios: set[str] = set()

        for record in records:
            if record.dataset_id != dataset.dataset_id:
                raise ValueError(
                    "All records must belong to the imported dataset."
                )

            if record.scenario_id in seen_scenarios:
                raise ValueError(
                    f"Duplicate scenario ID '{record.scenario_id}' "
                    "in dataset."
                )
            seen_scenarios.add(record.scenario_id)

            record_rows.append(
                {
                    "scenario_id": record.scenario_id,
                    "equipment": record.equipment,
                    "deviation": record.deviation,
                    "plant_unit": record.plant_unit,
                    "payload": self._json_payload(record),
                }
            )

        relation_rows = self._relation_rows(records)

        if dataset.record_count != len(record_rows):
            raise ValueError(
                "Dataset record_count does not match imported records."
            )
        if dataset.relationship_count != len(relation_rows):
            raise ValueError(
                "Dataset relationship_count does not match "
                "imported relationships."
            )

        remove_source, remove_target, set_source, set_target = (
            self._label_statements()
        )

        statement = (
            "CREATE (d:HazopDataset {"
            "dataset_id: $dataset_id, "
            "name: $name, "
            "filename: $filename, "
            "source_sha256: $source_sha256, "
            "record_count: $record_count, "
            "relationship_count: $relationship_count, "
            "created_at: $created_at"
            "}) "
            "CALL { "
            "WITH d "
            "UNWIND $records AS row "
            "CREATE (s:HazopScenario {"
            "dataset_id: $dataset_id, "
            "scenario_id: row.scenario_id, "
            "equipment: row.equipment, "
            "deviation: row.deviation, "
            "plant_unit: row.plant_unit, "
            "payload: row.payload"
            "}) "
            "CREATE (d)-[:CONTAINS_SCENARIO]->(s) "
            "RETURN count(s) AS records_created "
            "} "
            "CALL { "
            "WITH d "
            "UNWIND $relations AS row "
            "MERGE (source:HazopConcept "
            "{dataset_id: $dataset_id, key: row.subject_key}) "
            "SET source.name = row.subject "
            + remove_source
            + set_source
            + " MERGE (target:HazopConcept "
            "{dataset_id: $dataset_id, key: row.object_key}) "
            "SET target.name = row.object "
            + remove_target
            + set_target
            + " MERGE (source)-[edge:CAUSAL_EDGE "
            "{edge_id: row.edge_id}]->(target) "
            "SET edge.dataset_id = $dataset_id, "
            "edge.scenario_id = row.scenario_id, "
            "edge.predicate = row.predicate, "
            "edge.subject_type = row.subject_type, "
            "edge.object_type = row.object_type, "
            "edge.origin = row.origin, "
            "edge.evidence = row.evidence, "
            "edge.confidence = row.confidence, "
            "edge.source_id = row.source_id, "
            "edge.is_expert_verified = row.is_expert_verified, "
            "edge.source_location = row.source_location "
            "RETURN count(edge) AS relationships_created "
            "} "
            "RETURN d.dataset_id, records_created, relationships_created"
        )

        self._execute(
            [
                (
                    statement,
                    {
                        **dataset.to_dict(),
                        "records": record_rows,
                        "relations": relation_rows,
                    },
                )
            ]
        )

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        rows = self._execute(
            [
                (
                    (
                        "MATCH (d:HazopDataset {dataset_id: $dataset_id}) "
                        "RETURN properties(d)"
                    ),
                    {"dataset_id": dataset_id},
                )
            ]
        )[0]

        if not rows:
            return None
        if not rows[0]:
            raise Neo4jRepositoryError(
                "Invalid Neo4j response: missing dataset result column."
            )
        return self._dataset_from_properties(rows[0][0])

    def find_datasets(
        self,
        *,
        limit: int = 100,
    ) -> Sequence[DatasetMetadata]:
        self._validate_limit(limit)

        rows = self._execute(
            [
                (
                    (
                        "MATCH (d:HazopDataset) "
                        "RETURN properties(d) "
                        "ORDER BY d.created_at DESC, d.dataset_id "
                        "LIMIT $limit"
                    ),
                    {"limit": limit},
                )
            ]
        )[0]

        return [
            self._dataset_from_properties(row[0])
            for row in rows
            if row
        ]

    # ------------------------------------------------------------------
    # Record operations
    # ------------------------------------------------------------------

    def save_record(self, record: HazopRecord) -> None:
        """Upsert a record and synchronize its scenario-owned graph edges."""
        self._ensure_schema()

        relation_rows = self._relation_rows([record])
        remove_source, remove_target, set_source, set_target = (
            self._label_statements()
        )

        statement = (
            "MERGE (s:HazopScenario {"
            "dataset_id: $dataset_id, "
            "scenario_id: $scenario_id"
            "}) "
            "SET s.equipment = $equipment, "
            "s.deviation = $deviation, "
            "s.plant_unit = $plant_unit, "
            "s.payload = $payload "
            "WITH s "
            "OPTIONAL MATCH ()-[old:CAUSAL_EDGE {"
            "dataset_id: $dataset_id, "
            "scenario_id: $scenario_id"
            "}]->() "
            "DELETE old "
            "WITH s "
            "CALL { "
            "WITH s "
            "UNWIND $relations AS row "
            "MERGE (source:HazopConcept "
            "{dataset_id: $dataset_id, key: row.subject_key}) "
            "SET source.name = row.subject "
            + remove_source
            + set_source
            + " MERGE (target:HazopConcept "
            "{dataset_id: $dataset_id, key: row.object_key}) "
            "SET target.name = row.object "
            + remove_target
            + set_target
            + " MERGE (source)-[edge:CAUSAL_EDGE "
            "{edge_id: row.edge_id}]->(target) "
            "SET edge.dataset_id = $dataset_id, "
            "edge.scenario_id = row.scenario_id, "
            "edge.predicate = row.predicate, "
            "edge.subject_type = row.subject_type, "
            "edge.object_type = row.object_type, "
            "edge.origin = row.origin, "
            "edge.evidence = row.evidence, "
            "edge.confidence = row.confidence, "
            "edge.source_id = row.source_id, "
            "edge.is_expert_verified = row.is_expert_verified, "
            "edge.source_location = row.source_location "
            "RETURN count(edge) AS relationships_saved "
            "} "
            "RETURN s.scenario_id"
        )

        self._execute(
            [
                (
                    statement,
                    {
                        "dataset_id": record.dataset_id,
                        "scenario_id": record.scenario_id,
                        "equipment": record.equipment,
                        "deviation": record.deviation,
                        "plant_unit": record.plant_unit,
                        "payload": self._json_payload(record),
                        "relations": relation_rows,
                    },
                )
            ]
        )

    def get_record(
        self,
        scenario_id: str,
        *,
        dataset_id: str = "default",
    ) -> HazopRecord | None:
        rows = self._execute(
            [
                (
                    (
                        "MATCH (s:HazopScenario {"
                        "dataset_id: $dataset_id, "
                        "scenario_id: $scenario_id"
                        "}) "
                        "RETURN s.payload"
                    ),
                    {
                        "dataset_id": dataset_id,
                        "scenario_id": scenario_id,
                    },
                )
            ]
        )[0]

        if not rows:
            return None
        if not rows[0]:
            raise Neo4jRepositoryError(
                "Invalid Neo4j response: missing record payload column."
            )
        return self._record_from_payload(rows[0][0])

    def find_records(
        self,
        *,
        equipment: str | None = None,
        deviation: str | None = None,
        dataset_id: str | None = None,
        limit: int = 20,
    ) -> Sequence[HazopRecord]:
        self._validate_limit(limit)

        rows = self._execute(
            [
                (
                    (
                        "MATCH (s:HazopScenario) "
                        "WHERE ($dataset_id IS NULL "
                        "OR s.dataset_id = $dataset_id) "
                        "AND ($equipment IS NULL "
                        "OR toLower(s.equipment) = toLower($equipment)) "
                        "AND ($deviation IS NULL "
                        "OR toLower(s.deviation) = toLower($deviation)) "
                        "RETURN s.payload "
                        "ORDER BY s.dataset_id, s.scenario_id "
                        "LIMIT $limit"
                    ),
                    {
                        "dataset_id": dataset_id,
                        "equipment": equipment,
                        "deviation": deviation,
                        "limit": limit,
                    },
                )
            ]
        )[0]

        records: list[HazopRecord] = []
        for row in rows:
            if not row:
                raise Neo4jRepositoryError(
                    "Invalid Neo4j response: missing record payload column."
                )
            records.append(self._record_from_payload(row[0]))
        return records

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def save_relations(
        self,
        relations: Sequence[CausalRelation],
    ) -> None:
        """Upsert standalone relationships without collapsing evidence."""
        self._ensure_schema()

        grouped: dict[str, list[dict[str, object]]] = {}
        for row, relation in zip(
            self._standalone_relation_rows(relations),
            relations,
            strict=True,
        ):
            grouped.setdefault(relation.dataset_id, []).append(row)

        for dataset_id, rows in grouped.items():
            self._upsert_relation_rows(dataset_id, rows)

    def find_relations(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        dataset_id: str | None = None,
        limit: int = 50,
    ) -> Sequence[CausalRelation]:
        self._validate_limit(limit)

        rows = self._execute(
            [
                (
                    (
                        "MATCH (source:HazopConcept)"
                        "-[edge:CAUSAL_EDGE]->"
                        "(target:HazopConcept) "
                        "WHERE ($dataset_id IS NULL "
                        "OR edge.dataset_id = $dataset_id) "
                        "AND ($subject IS NULL "
                        "OR toLower(source.name) = toLower($subject)) "
                        "AND ($predicate IS NULL "
                        "OR toLower(edge.predicate) = toLower($predicate)) "
                        "AND ($object IS NULL "
                        "OR toLower(target.name) = toLower($object)) "
                        "RETURN properties(edge), source.name, target.name "
                        "ORDER BY edge.dataset_id, edge.scenario_id, "
                        "edge.predicate, edge.edge_id "
                        "LIMIT $limit"
                    ),
                    {
                        "dataset_id": dataset_id,
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                        "limit": limit,
                    },
                )
            ]
        )[0]

        results: list[CausalRelation] = []
        for row in rows:
            if len(row) != 3:
                raise Neo4jRepositoryError(
                    "Invalid Neo4j response: relationship row must "
                    "contain three columns."
                )
            results.append(
                self._relation_from_properties(row[0], row[1], row[2])
            )
        return results

    # ------------------------------------------------------------------
    # Shared validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be greater than zero")
