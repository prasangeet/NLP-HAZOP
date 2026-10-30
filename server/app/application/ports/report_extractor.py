"""Port for extracting typed HAZOP propositions from unstructured reports."""
from __future__ import annotations
from typing import Protocol
from app.domain.entities.extracted_report import ExtractedReport

class ReportExtractor(Protocol):
    def extract(self, text: str, *, title: str) -> ExtractedReport:
        """Extract structured HAZOP scenarios and source-quoted relationships."""
        ...
