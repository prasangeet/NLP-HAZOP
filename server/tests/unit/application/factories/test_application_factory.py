"""Tests for factory composition and strategy selection."""
from app.application.factories.application_factory import ApplicationFactory
from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.domain.reasoning.models import ReasoningStatus, ReasoningResult, AbstentionReason
from app.infrastructure.graph.in_memory_graph_repository import InMemoryGraphRepository
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory

class FakeLLMReasoner:
    def reason(self, context):
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=context.scenario_id,
            dataset_id=context.dataset_id,
            abstention_reason=AbstentionReason.NO_EVIDENCE,
        )

class FakeAIProviderFactory(AIProviderFactory):
    def create_report_extractor(self):
        raise NotImplementedError
    def create_missing_link_reasoner(self):
        return FakeLLMReasoner()

def test_application_factory_selects_graph_llm_and_hybrid_strategies():
    factory = ApplicationFactory(InMemoryGraphRepository(), FakeAIProviderFactory())
    assert isinstance(factory.create_reasoning_strategy("graph"), GraphReasoner)
    assert isinstance(factory.create_reasoning_strategy("llm"), FakeLLMReasoner)
    assert isinstance(factory.create_reasoning_strategy("hybrid"), HybridReasoner)

def test_llm_strategies_require_provider_factory():
    factory = ApplicationFactory(InMemoryGraphRepository())
    try:
        factory.create_reasoning_strategy("hybrid")
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected LLM configuration error")
