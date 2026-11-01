"""Regression tests for Groq GPT-OSS structured-output configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from app.application.strategies import llm_reasoner, prompt_only_reasoner
from app.infrastructure.extraction import groq_report_extractor


class FakeChatGroq:
    """Capture the provider configuration without making a network request."""

    instances: ClassVar[list[FakeChatGroq]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.structured_schema: Any = None
        self.structured_kwargs: dict[str, Any] = {}
        self.instances.append(self)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeChatGroq:
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return self


def install_fake(monkeypatch: Any) -> None:
    FakeChatGroq.instances.clear()
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_reasoner, "ChatGroq", FakeChatGroq)
    monkeypatch.setattr(prompt_only_reasoner, "ChatGroq", FakeChatGroq)
    monkeypatch.setattr(groq_report_extractor, "ChatGroq", FakeChatGroq)


def assert_gpt_oss_json_schema(model: FakeChatGroq) -> None:
    assert model.kwargs["model"] == "openai/gpt-oss-120b"
    assert model.structured_kwargs == {
        "method": "json_schema",
        "strict": False,
    }
    assert model.structured_schema is not None


def test_evidence_reasoner_uses_groq_json_schema(monkeypatch: Any) -> None:
    install_fake(monkeypatch)

    llm_reasoner.LLMReasoner()

    assert_gpt_oss_json_schema(FakeChatGroq.instances[-1])


def test_prompt_only_reasoner_uses_groq_json_schema(monkeypatch: Any) -> None:
    install_fake(monkeypatch)

    prompt_only_reasoner.PromptOnlyReasoner()._get_chat_model()

    assert_gpt_oss_json_schema(FakeChatGroq.instances[-1])


def test_report_extractor_uses_groq_json_schema(monkeypatch: Any) -> None:
    install_fake(monkeypatch)

    groq_report_extractor.GroqReportExtractor()

    assert_gpt_oss_json_schema(FakeChatGroq.instances[-1])
