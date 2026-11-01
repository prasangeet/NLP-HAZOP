"""Factory Method composition root for HAZOP application workflows."""
from __future__ import annotations
from typing import Literal
from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.application.strategies.prompt_only_reasoner import PromptOnlyReasoner
from app.application.strategies.retrieval.graph_neighborhood import GraphNeighborhoodRetriever
from app.application.strategies.retrieval.hybrid import HybridRetriever
from app.application.strategies.retrieval.scenario_local import ScenarioLocalRetriever
from app.application.strategies.retrieval.tfidf import TfidfRetriever
from app.application.strategies.retrieval.models import RetrievalStrategy
from app.domain.repositories.graph_repository import DatasetRepository
from app.domain.reasoning.models import ReasoningStrategy
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory
RecoveryMode = Literal["graph", "llm", "hybrid"]
class ApplicationFactory:
    """Build complete use-case dependencies in one place."""
    def __init__(self, repository: DatasetRepository, ai_factory: AIProviderFactory | None = None):
        self._repository = repository
        self._ai_factory = ai_factory
    def create_retrieval_strategy(self) -> RetrievalStrategy:
        local = ScenarioLocalRetriever(self._repository)
        vector = TfidfRetriever(self._repository)
        return GraphNeighborhoodRetriever(HybridRetriever(local, vector), self._repository)
    def create_reasoning_strategy(self, mode: RecoveryMode) -> ReasoningStrategy:
        if mode == "graph":
            return GraphReasoner()
        if self._ai_factory is None:
            raise RuntimeError("GROQ_API_KEY is required for LLM and hybrid strategies.")
        if mode == "llm":
            return self._ai_factory.create_missing_link_reasoner()
        if mode == "hybrid":
            return HybridReasoner(
                GraphReasoner(),
                self._ai_factory.create_missing_link_reasoner(),
                fallback_reasoner=PromptOnlyReasoner(include_graph_context=True),
            )
        raise ValueError(f"Unsupported recovery mode: {mode}")
