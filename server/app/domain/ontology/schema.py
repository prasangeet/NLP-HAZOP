"""Canonical typed HAZOP ontology used by extraction, storage and validation.

The schema is intentionally small and explicit. It prevents the repository from
silently treating every string as an untyped concept and gives graph traversal
a stable vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    PROCESS_UNIT = "ProcessUnit"
    EQUIPMENT = "Equipment"
    PARAMETER = "Parameter"
    DEVIATION = "Deviation"
    CAUSE = "Cause"
    EFFECT = "Effect"
    CONSEQUENCE = "Consequence"
    SAFEGUARD = "Safeguard"
    RECOMMENDATION = "Recommendation"
    CONCEPT = "Concept"


ENTITY_LABELS = tuple(
    item.value for item in EntityType if item != EntityType.CONCEPT
)


@dataclass(frozen=True, slots=True)
class RelationSpec:
    """Allowed source and target entity types for one relationship predicate."""

    subject_types: frozenset[EntityType]
    object_types: frozenset[EntityType]


RELATION_SCHEMA: dict[str, RelationSpec] = {
    "causes_deviation": RelationSpec(
        frozenset({EntityType.CAUSE}),
        frozenset({EntityType.DEVIATION}),
    ),
    "leads_to": RelationSpec(
        frozenset({EntityType.DEVIATION}),
        frozenset({EntityType.EFFECT}),
    ),
    "leads_to_consequence": RelationSpec(
        frozenset({EntityType.EFFECT, EntityType.DEVIATION}),
        frozenset({EntityType.CONSEQUENCE}),
    ),
    "co_occurs_with_consequence": RelationSpec(
        frozenset({EntityType.DEVIATION}),
        frozenset({EntityType.CONSEQUENCE}),
    ),
    # Benchmark output label: a cause and consequence co-listed in one
    # reconstructed worksheet row. This is deliberately not a causal edge.
    "co_listed_with_consequence": RelationSpec(
        frozenset({EntityType.CAUSE}),
        frozenset({EntityType.CONSEQUENCE}),
    ),
    "mitigated_by": RelationSpec(
        frozenset({
            EntityType.DEVIATION,
            EntityType.EFFECT,
            EntityType.CONSEQUENCE,
        }),
        frozenset({EntityType.SAFEGUARD}),
    ),
    "recommends_action": RelationSpec(
        frozenset({EntityType.DEVIATION, EntityType.CONSEQUENCE}),
        frozenset({EntityType.RECOMMENDATION}),
    ),
    "has_parameter": RelationSpec(
        frozenset({EntityType.EQUIPMENT}),
        frozenset({EntityType.PARAMETER}),
    ),
    "located_in": RelationSpec(
        frozenset({EntityType.EQUIPMENT}),
        frozenset({EntityType.PROCESS_UNIT}),
    ),
}


def infer_entity_types(
    predicate: str,
    subject: str,
    object_value: str,
    *,
    deviation: str = "",
) -> tuple[EntityType, EntityType]:
    """Infer endpoint types for legacy records with no explicit type metadata."""
    normalized = predicate.strip().casefold()
    if normalized == "causes_deviation":
        return EntityType.CAUSE, EntityType.DEVIATION
    if normalized == "leads_to":
        return EntityType.DEVIATION, EntityType.EFFECT
    if normalized == "leads_to_consequence":
        if deviation and subject.strip().casefold() == deviation.strip().casefold():
            return EntityType.DEVIATION, EntityType.CONSEQUENCE
        return EntityType.EFFECT, EntityType.CONSEQUENCE
    if normalized == "co_occurs_with_consequence":
        return EntityType.DEVIATION, EntityType.CONSEQUENCE
    if normalized == "co_listed_with_consequence":
        return EntityType.CAUSE, EntityType.CONSEQUENCE
    if normalized == "mitigated_by":
        return EntityType.CONSEQUENCE, EntityType.SAFEGUARD
    if normalized == "recommends_action":
        return EntityType.CONSEQUENCE, EntityType.RECOMMENDATION
    if normalized == "has_parameter":
        return EntityType.EQUIPMENT, EntityType.PARAMETER
    if normalized == "located_in":
        return EntityType.EQUIPMENT, EntityType.PROCESS_UNIT
    return EntityType.CONCEPT, EntityType.CONCEPT


def validate_relation_types(
    predicate: str,
    subject_type: EntityType,
    object_type: EntityType,
) -> str | None:
    """Return a schema violation message, or None when the edge is well typed."""
    spec = RELATION_SCHEMA.get(predicate.strip().casefold())
    if spec is None:
        return f"Unsupported relationship predicate: {predicate!r}."
    if subject_type not in spec.subject_types:
        expected = ", ".join(sorted(item.value for item in spec.subject_types))
        return (
            f"{predicate!r} expects a subject of type {expected}; "
            f"received {subject_type.value}."
        )
    if object_type not in spec.object_types:
        expected = ", ".join(sorted(item.value for item in spec.object_types))
        return (
            f"{predicate!r} expects an object of type {expected}; "
            f"received {object_type.value}."
        )
    return None
