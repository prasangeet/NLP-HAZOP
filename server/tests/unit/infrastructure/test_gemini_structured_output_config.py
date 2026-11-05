"""Verify every structured-output adapter uses the fixed Gemini model."""

from __future__ import annotations

from typing import Any, ClassVar

from app.application.strategies import llm_reasoner, prompt_only_reasoner
from app.infrastructure.extraction import gemini_report_extractor


class FakeChatGoogleGenerativeAI:
    """Capture Gemini model configuration without making network calls."""

    instances: ClassVar[list["FakeChatGoogleGenerativeAI"]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.schema: Any = None
        self.structured_kwargs: dict[str, Any] = {}
        self.instances.append(self)

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> "FakeChatGoogleGenerativeAI":
        self.schema = schema
        self.structured_kwargs = kwargs
        return self


def test_structured_output_adapters_use_fixed_gemini_model(
    monkeypatch: Any,
) -> None:
    FakeChatGoogleGenerativeAI.instances.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        llm_reasoner, "ChatGoogleGenerativeAI", FakeChatGoogleGenerativeAI
    )
    monkeypatch.setattr(
        prompt_only_reasoner,
        "ChatGoogleGenerativeAI",
        FakeChatGoogleGenerativeAI,
    )
    monkeypatch.setattr(
        gemini_report_extractor,
        "ChatGoogleGenerativeAI",
        FakeChatGoogleGenerativeAI,
    )

    llm_reasoner.LLMReasoner()
    prompt_only_reasoner.PromptOnlyReasoner()._get_chat_model()
    gemini_report_extractor.GeminiReportExtractor()

    assert len(FakeChatGoogleGenerativeAI.instances) == 4
    assert all(
        instance.kwargs["model"] == "gemini-3.5-flash-lite"
        for instance in FakeChatGoogleGenerativeAI.instances
    )
