"""
Domain entities for structured HAZOP records.

These models are independent of databases, APIs, and infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from app.domain.ontology.schema import EntityType


class RecordType(StrEnum):
    """Origin or nature of a HAZOP record."""

    SOURCE_VERIFIED = "source_verified"
    SOURCE_RECONSTRUCTED = "source_reconstructed"
    SYNTHETIC = "synthetic"
    EXPERT_VALIDATED = "expert_validated"
    UNSPECIFIED = "unspecified"


class ValidationStatus(StrEnum):
    """Validation state of a HAZOP record."""

    UNREVIEWED = "unreviewed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class RelationshipOrigin(StrEnum):
    """Origin of a causal or semantic relationship."""

    SOURCE = "source"
    RECONSTRUCTED = "reconstructed"
    SYNTHETIC = "synthetic"
    EXPERT = "expert"
    UNSPECIFIED = "unspecified"
    MODEL_INFERRED = "model_inferred"
    GRAPH_INFERRED = "graph_inferred"
    RETRIEVAL_INFERRED = "retrieval_inferred"
    LLM_EXTRACTED = "llm_extracted"


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Reference to the document or evidence supporting a record."""

    title: str
    document_id: str | None = None
    location: str | None = None
    provenance_note: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("SourceReference.title must not be empty.")

        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(
            self,
            "document_id",
            self.document_id.strip() or None
            if self.document_id is not None
            else None,
        )
        object.__setattr__(
            self,
            "location",
            self.location.strip() or None
            if self.location is not None
            else None,
        )
        object.__setattr__(
            self, "provenance_note", self.provenance_note.strip()
        )


@dataclass(frozen=True, slots=True)
class CausalRelation:
    """A directed relationship between two HAZOP concepts."""

    subject: str
    predicate: str
    object: str
    origin: RelationshipOrigin
    evidence: str = ""
    confidence: float | None = None
    source_id: str | None = None
    is_expert_verified: bool = False
    dataset_id: str = "default"
    subject_type: EntityType = EntityType.CONCEPT
    object_type: EntityType = EntityType.CONCEPT

    def __post_init__(self) -> None:
        for name in ("subject", "predicate", "object"):
            value = getattr(self, name).strip()

            if not value:
                raise ValueError(
                    f"CausalRelation.{name} must not be empty."
                )

            object.__setattr__(self, name, value)

        object.__setattr__(self, "evidence", self.evidence.strip())
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("CausalRelation.dataset_id must not be empty.")
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())

        if self.source_id is not None:
            object.__setattr__(
                self, "source_id", self.source_id.strip() or None
            )

        if self.confidence is not None and not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "CausalRelation.confidence must be between 0.0 and 1.0."
            )

        for field_name in ("subject_type", "object_type"):
            value = getattr(self, field_name)
            try:
                normalized_type = value if isinstance(value, EntityType) else EntityType(value)
            except ValueError as exc:
                raise ValueError(f"CausalRelation.{field_name} is not a supported HAZOP entity type.") from exc
            object.__setattr__(self, field_name, normalized_type)

        if self.is_expert_verified and self.origin != RelationshipOrigin.EXPERT:
            object.__setattr__(self, "is_expert_verified", False)


@dataclass(frozen=True, slots=True)
class HazopRecord:
    """Normalized HAZOP scenario with provenance and causal relations."""

    scenario_id: str
    equipment: str
    deviation: str

    plant_unit: str = ""
    design_intent: str = ""
    parameter: str = ""
    guideword: str = ""

    causes: tuple[str, ...] = field(default_factory=tuple)
    effects: tuple[str, ...] = field(default_factory=tuple)
    consequences: tuple[str, ...] = field(default_factory=tuple)
    safeguards: tuple[str, ...] = field(default_factory=tuple)
    recommendations: tuple[str, ...] = field(default_factory=tuple)

    record_type: RecordType = RecordType.UNSPECIFIED
    validation_status: ValidationStatus = ValidationStatus.UNREVIEWED

    source: SourceReference | None = None
    causal_relations: tuple[CausalRelation, ...] = field(
        default_factory=tuple
    )

    risk_ranking: str | None = None
    remarks: str = ""
    row_number: int | None = None
    raw_row: dict[str, object] = field(default_factory=dict)
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("HazopRecord.dataset_id must not be empty.")
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        for name in ("scenario_id", "equipment", "deviation"):
            value = getattr(self, name).strip()
            if not value:
                raise ValueError(f"HazopRecord.{name} must not be empty.")
            object.__setattr__(self, name, value)

        for name in (
            "plant_unit",
            "design_intent",
            "parameter",
            "guideword",
            "remarks",
        ):
            object.__setattr__(self, name, getattr(self, name).strip())

        for name in (
            "causes",
            "effects",
            "consequences",
            "safeguards",
            "recommendations",
        ):
            normalized: list[str] = []
            seen: set[str] = set()
            for item in getattr(self, name):
                item = item.strip()
                key = item.casefold()
                if item and key not in seen:
                    normalized.append(item)
                    seen.add(key)
            object.__setattr__(self, name, tuple(normalized))

        if self.risk_ranking is not None:
            cleaned_risk = self.risk_ranking.strip()
            if (
                not cleaned_risk
                or cleaned_risk.casefold() in {"", "-", "—", "–", "n/a", "na", "none", "not stated", "null"}
                or all(char in "-—–_ \t" for char in cleaned_risk)
            ):
                object.__setattr__(self, "risk_ranking", None)
            else:
                object.__setattr__(self, "risk_ranking", cleaned_risk)

        normalized_relations = []
        for relation in self.causal_relations:
            if relation.dataset_id not in {"default", self.dataset_id}:
                raise ValueError(
                    "Causal relation dataset_id must match its HAZOP record."
                )
            if relation.dataset_id == "default" and self.dataset_id != "default":
                relation = replace(relation, dataset_id=self.dataset_id)
            normalized_relations.append(relation)
        object.__setattr__(self, "causal_relations", tuple(normalized_relations))
        object.__setattr__(
            self, "raw_row", dict(self.raw_row)
        )

    @property
    def has_source_evidence(self) -> bool:
        """Return whether a source reference is attached."""
        return self.source is not None

    def to_dict(self) -> dict[str, object]:
        """Serialize the record into a JSON-friendly dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "plant_unit": self.plant_unit,
            "equipment": self.equipment,
            "design_intent": self.design_intent,
            "guideword": self.guideword,
            "parameter": self.parameter,
            "deviation": self.deviation,
            "causes": list(self.causes),
            "effects": list(self.effects),
            "consequences": list(self.consequences),
            "safeguards": list(self.safeguards),
            "recommendations": list(self.recommendations),
            "record_type": self.record_type.value,
            "validation_status": self.validation_status.value,
            "source": (
                {
                    "title": self.source.title,
                    "document_id": self.source.document_id,
                    "location": self.source.location,
                    "provenance_note": self.source.provenance_note,
                }
                if self.source is not None
                else None
            ),
            "causal_relations": [
                {
                    "subject": relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "confidence": relation.confidence,
                    "source_id": relation.source_id,
                    "is_expert_verified": relation.is_expert_verified,
                    "dataset_id": relation.dataset_id,
                    "subject_type": relation.subject_type.value,
                    "object_type": relation.object_type.value,
                }
                for relation in self.causal_relations
            ],
            "risk_ranking": self.risk_ranking,
            "remarks": self.remarks,
            "row_number": self.row_number,
            "raw_row": dict(self.raw_row),
        }
