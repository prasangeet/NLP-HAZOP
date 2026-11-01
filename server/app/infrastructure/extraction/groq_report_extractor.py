"""Groq adapter implementing source-grounded HAZOP report extraction."""
from __future__ import annotations
import os
from typing import Protocol
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr
from app.domain.entities.extracted_report import ExtractedRelationship, ExtractedReport, ExtractedScenario
from app.domain.ontology.schema import EntityType

class StructuredExtractorModel(Protocol):
    def invoke(self, prompt: str) -> object: ...

class _RelationshipOutput(BaseModel):
    subject: str
    predicate: str
    object: str
    subject_type: EntityType
    object_type: EntityType
    evidence_quote: str = Field(description="Verbatim substring copied from the supplied report.")
    source_location: str = ""

class _ScenarioOutput(BaseModel):
    scenario_id: str
    plant_unit: str = ""
    equipment: str
    design_intent: str = ""
    guideword: str = ""
    parameter: str = ""
    deviation: str
    causes: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    safeguards: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    relationships: list[_RelationshipOutput] = Field(default_factory=list)
    source_location: str = ""

class _ReportOutput(BaseModel):
    scenarios: list[_ScenarioOutput] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)

class GroqReportExtractor:
    """Extract scenarios and typed causal claims with exact source quotes."""
    def __init__(self, model_name: str = "openai/gpt-oss-120b", *, chat_model: StructuredExtractorModel | None = None):
        if chat_model is not None:
            self._model = chat_model
            return
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for report extraction.")
        self._model = ChatGroq(model=model_name, temperature=0.0, api_key=SecretStr(api_key)).with_structured_output(_ReportOutput, method="json_schema", strict=False)

    def extract(self, text: str, *, title: str) -> ExtractedReport:
        if not text.strip():
            raise ValueError("The supplied report is empty.")
        prompt = f"""
You extract structured HAZOP knowledge from a historical engineering report.
Return only facts explicitly supported by the supplied text. Never invent
equipment tags, process conditions, causes, consequences, safeguards, scenario
identifiers, or causal relationships. Use empty strings/lists for missing data.
For every relationship, use a canonical predicate:
causes_deviation, leads_to, leads_to_consequence, co_occurs_with_consequence,
mitigated_by, recommends_action, has_parameter, located_in.
Use entity types: ProcessUnit, Equipment, Parameter, Deviation, Cause, Effect,
Consequence, Safeguard, Recommendation, Concept.
evidence_quote MUST be copied verbatim from the source, including punctuation.
Include page/section/paragraph location when available. Separate each scenario.
A proposition extracted from a report is not proof of physical causality.
If no scenario can be extracted, return an empty scenario list.
Title: {title}
SOURCE TEXT:
{text}
""".strip()
        response = self._model.invoke(prompt)
        output = response if isinstance(response, _ReportOutput) else _ReportOutput.model_validate(response)
        scenarios = tuple(
            ExtractedScenario(
                scenario_id=item.scenario_id, plant_unit=item.plant_unit,
                equipment=item.equipment, design_intent=item.design_intent,
                guideword=item.guideword, parameter=item.parameter,
                deviation=item.deviation, causes=tuple(item.causes),
                effects=tuple(item.effects), consequences=tuple(item.consequences),
                safeguards=tuple(item.safeguards), recommendations=tuple(item.recommendations),
                relationships=tuple(ExtractedRelationship(
                    subject=r.subject, predicate=r.predicate, object=r.object,
                    subject_type=r.subject_type, object_type=r.object_type,
                    evidence_quote=r.evidence_quote, source_location=r.source_location,
                ) for r in item.relationships),
                source_location=item.source_location,
            ) for item in output.scenarios
        )
        return ExtractedReport(title=title, scenarios=scenarios, extraction_notes=tuple(output.extraction_notes))
