"""Source-grounded report extraction results."""
from __future__ import annotations
from dataclasses import dataclass
from app.domain.ontology.schema import EntityType

@dataclass(frozen=True, slots=True)
class ExtractedRelationship:
    subject: str
    predicate: str
    object: str
    subject_type: EntityType
    object_type: EntityType
    evidence_quote: str
    source_location: str = ""
    def __post_init__(self) -> None:
        for name in ("subject", "predicate", "object", "evidence_quote"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        for name in ("subject_type", "object_type"):
            value = getattr(self, name)
            if not isinstance(value, EntityType):
                object.__setattr__(self, name, EntityType(value))
        object.__setattr__(self, "source_location", self.source_location.strip())

@dataclass(frozen=True, slots=True)
class ExtractedScenario:
    scenario_id: str
    equipment: str
    deviation: str
    plant_unit: str = ""
    design_intent: str = ""
    guideword: str = ""
    parameter: str = ""
    causes: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    safeguards: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    relationships: tuple[ExtractedRelationship, ...] = ()
    source_location: str = ""
    def __post_init__(self) -> None:
        for name in ("scenario_id", "equipment", "deviation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        for name in ("plant_unit", "design_intent", "guideword", "parameter", "source_location"):
            object.__setattr__(self, name, getattr(self, name).strip())
        for name in ("causes", "effects", "consequences", "safeguards", "recommendations", "relationships"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

@dataclass(frozen=True, slots=True)
class ExtractedReport:
    title: str
    scenarios: tuple[ExtractedScenario, ...]
    extraction_notes: tuple[str, ...] = ()
    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Report title must not be empty")
        if not self.scenarios:
            raise ValueError("The report contains no extractable HAZOP scenarios")
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "extraction_notes", tuple(self.extraction_notes))
