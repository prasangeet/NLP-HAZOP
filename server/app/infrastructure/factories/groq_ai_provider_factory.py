"""Concrete Groq Abstract Factory for Llama-based extraction and reasoning."""
from app.application.ports.report_extractor import ReportExtractor
from app.application.strategies.llm_reasoner import LLMReasoner
from app.domain.reasoning.models import ReasoningStrategy
from app.infrastructure.extraction.groq_report_extractor import GroqReportExtractor
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory
class GroqAIProviderFactory(AIProviderFactory):
    """Create Groq-backed report extractor and evidence reasoner."""
    def create_report_extractor(self) -> ReportExtractor:
        return GroqReportExtractor()
    def create_missing_link_reasoner(self) -> ReasoningStrategy:
        return LLMReasoner()
