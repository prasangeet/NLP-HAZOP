"""Formal HAZOP ontology vocabulary and relationship constraints."""

from app.domain.ontology.schema import (
    ENTITY_LABELS,
    RELATION_SCHEMA,
    EntityType,
    RelationSpec,
    infer_entity_types,
    validate_relation_types,
)

__all__ = [
    "ENTITY_LABELS",
    "RELATION_SCHEMA",
    "EntityType",
    "RelationSpec",
    "infer_entity_types",
    "validate_relation_types",
]
