
"""Gemini provider factory for HAZOP AI components."""

from app.application.ports.report_extractor import ReportExtractor
from app.application.strategies.llm_reasoner import LLMReasoner
from app.domain.reasoning.models import ReasoningStrategy
from app.infrastructure.extraction.gemini_report_extractor import (
    GeminiReportExtractor,
)
from app.infrastructure.factories.ai_provider_factory import (
    AIProviderFactory,
)


class GeminiAIProviderFactory(AIProviderFactory):
    """Create Gemini-based report extraction and reasoning components."""

    def create_report_extractor(self) -> ReportExtractor:
        """Create the Gemini report extractor."""
        return GeminiReportExtractor()

    def create_missing_link_reasoner(self) -> ReasoningStrategy:
        """Create the Gemini missing-link reasoner."""
        return LLMReasoner()
