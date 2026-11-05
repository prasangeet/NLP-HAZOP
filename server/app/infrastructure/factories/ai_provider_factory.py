"""Abstract factory for interchangeable AI provider components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.application.ports.report_extractor import ReportExtractor
from app.domain.reasoning.models import ReasoningStrategy


class AIProviderFactory(ABC):
    """Create a compatible family of extraction and reasoning adapters."""

    @abstractmethod
    def create_report_extractor(self) -> ReportExtractor:
        """Create a report extractor."""

    @abstractmethod
    def create_missing_link_reasoner(self) -> ReasoningStrategy:
        """Create an evidence-grounded missing-link reasoner."""

    def create_fallback_reasoner(self) -> ReasoningStrategy | None:
        """Create an optional scenario-only fallback reasoner."""

        return None
