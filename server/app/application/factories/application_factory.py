"""Factory Method composition root for HAZOP application workflows."""

from __future__ import annotations

from typing import Literal

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.application.strategies.retrieval.graph_neighborhood import (
    GraphNeighborhoodRetriever,
)
from app.application.strategies.retrieval.hybrid import HybridRetriever
from app.application.strategies.retrieval.models import RetrievalStrategy
from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.application.strategies.retrieval.tfidf import TfidfRetriever
from app.domain.reasoning.models import ReasoningStrategy
from app.domain.repositories.graph_repository import DatasetRepository
from app.infrastructure.factories.ai_provider_factory import AIProviderFactory


RecoveryMode = Literal["graph", "llm", "hybrid"]


class ApplicationFactory:
    """Compose application strategies from infrastructure dependencies."""

    def __init__(
        self,
        repository: DatasetRepository,
        ai_factory: AIProviderFactory | None = None,
    ) -> None:
        self._repository = repository
        self._ai_factory = ai_factory

    def create_retrieval_strategy(self) -> RetrievalStrategy:
        """Create the graph-aware retrieval pipeline."""

        local_retriever = ScenarioLocalRetriever(self._repository)
        vector_retriever = TfidfRetriever(self._repository)

        hybrid_retriever = HybridRetriever(
            local_retriever,
            vector_retriever,
        )

        return GraphNeighborhoodRetriever(
            hybrid_retriever,
            self._repository,
        )

    def create_reasoning_strategy(
        self,
        mode: RecoveryMode,
    ) -> ReasoningStrategy:
        """Create the requested graph, LLM, or hybrid reasoning strategy."""

        if mode not in ("graph", "llm", "hybrid"):
            raise ValueError(f"Unsupported reasoning mode: {mode}")

        if mode == "graph":
            return GraphReasoner()

        if self._ai_factory is None:
            raise RuntimeError(
                "GEMINI_API_KEY is required for LLM and hybrid strategies."
            )

        llm_reasoner = self._ai_factory.create_missing_link_reasoner()
        fallback_reasoner = self._ai_factory.create_fallback_reasoner()

        if mode == "llm":
            return llm_reasoner

        return HybridReasoner(
            graph_reasoner=GraphReasoner(),
            llm_reasoner=llm_reasoner,
            fallback_reasoner=fallback_reasoner,
        )
