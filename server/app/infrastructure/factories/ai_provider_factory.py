"""Abstract Factory for interchangeable AI provider components."""
from __future__ import annotations
from abc import ABC, abstractmethod
from app.application.ports.report_extractor import ReportExtractor
from app.domain.reasoning.models import ReasoningStrategy
class AIProviderFactory(ABC):
    """Create a compatible family of extraction and reasoning adapters."""
    @abstractmethod
    def create_report_extractor(self) -> ReportExtractor:
        raise NotImplementedError
    @abstractmethod
    def create_missing_link_reasoner(self) -> ReasoningStrategy:
        raise NotImplementedError
