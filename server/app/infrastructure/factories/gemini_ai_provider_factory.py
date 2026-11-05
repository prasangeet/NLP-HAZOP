
"""Gemini provider factory for HAZOP AI components."""

import os
from typing import cast

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.application.ports.report_extractor import ReportExtractor
from app.application.strategies.llm_reasoner import LLMReasoner
from app.application.strategies.prompt_only_reasoner import (
    PromptOnlyDecision,
    PromptOnlyReasoner,
)
from app.domain.reasoning.models import ReasoningStrategy
from app.infrastructure.extraction.gemini_report_extractor import (
    GeminiReportExtractor,
)
from app.infrastructure.factories.ai_provider_factory import (
    AIProviderFactory,
)

_MODEL = "gemini-3.5-flash-lite"


class GeminiAIProviderFactory(AIProviderFactory):
    """Create Gemini-based report extraction and reasoning components."""

    def create_report_extractor(self) -> ReportExtractor:
        """Create the Gemini report extractor."""
        return GeminiReportExtractor()

    def create_missing_link_reasoner(self) -> ReasoningStrategy:
        """Create the Gemini missing-link reasoner."""
        return LLMReasoner()

    def create_fallback_reasoner(self) -> ReasoningStrategy | None:
        """Create a Gemini-backed prompt-only fallback reasoner.

        This generates an independent hypothesis from scenario context alone
        when evidence-backed Gemini reasoning cannot resolve ambiguity.
        Returns None if GEMINI_API_KEY is not configured.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None

        gemini_model = ChatGoogleGenerativeAI(
            model=_MODEL,
            temperature=0.0,
            google_api_key=SecretStr(api_key),
        ).with_structured_output(PromptOnlyDecision)

        return PromptOnlyReasoner(
            chat_model=cast(object, gemini_model),
            include_graph_context=True,
        )
