"""Contracts shared by HAZOP evidence retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.domain.entities.hazop_record import CausalRelation


class RetrievalScope(StrEnum):
    """Scope from which evidence was retrieved."""

    SCENARIO_LOCAL = "scenario_local"
    CROSS_SCENARIO = "cross_scenario"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """A relationship retrieved with its scenario provenance."""

    scenario_id: str
    relation: CausalRelation
    scope: RetrievalScope
    relevance_score: float | None = None
    dataset_id: str = "default"


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Input contract for retrieving evidence for a partial causal link."""

    scenario_id: str
    predicate: str
    known_endpoint: str
    missing_endpoint: str
    limit: int = 20
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_id",
            "dataset_id",
            "predicate",
            "known_endpoint",
            "missing_endpoint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string"
                )
            object.__setattr__(self, field_name, value.strip())

        if self.missing_endpoint not in {"subject", "object"}:
            raise ValueError(
                "MissingEndpoint must be either 'subject' or 'object'"
            )

        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Retrieved evidence and graph context for the target scenario."""

    scenario_id: str
    scope: RetrievalScope
    evidence: tuple[RetrievedEvidence, ...] = ()
    scenario_exists: bool = True
    graph_context: tuple[RetrievedEvidence, ...] = ()
    scenario_context: str = ""
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "graph_context", tuple(self.graph_context))

    @property
    def is_empty(self) -> bool:
        return not self.evidence


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Interface for interchangeable evidence retrieval methods."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve evidence relevant to a partial causal link."""
        ...
