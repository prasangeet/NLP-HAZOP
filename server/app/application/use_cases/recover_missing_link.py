"""Recover missing links from retrieved HAZOP evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalScope,
    RetrievalStrategy,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import CausalRelation


class MissingEndpoint(StrEnum):
    SUBJECT = "subject"
    OBJECT = "object"


class RecoveryStatus(StrEnum):
    CANDIDATES_FOUND = "candidates_found"
    NO_CANDIDATES = "no_candidates"
    SCENARIO_NOT_FOUND = "scenario_not_found"


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    scenario_id: str
    predicate: str
    known_endpoint: str
    missing_endpoint: MissingEndpoint
    limit: int = 20
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        for field_name in ("scenario_id", "dataset_id", "predicate", "known_endpoint"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        if not isinstance(self.missing_endpoint, MissingEndpoint):
            try:
                object.__setattr__(
                    self,
                    "missing_endpoint",
                    MissingEndpoint(self.missing_endpoint),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "MissingEndpoint must be either 'subject' or 'object'"
                ) from exc

        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("limit must be an integer")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    """A recovered candidate preserving the original relation object."""

    missing_value: str
    relation: CausalRelation
    scenario_id: str = ""
    scope: RetrievalScope = RetrievalScope.SCENARIO_LOCAL
    relevance_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.missing_value, str) or not self.missing_value.strip():
            raise ValueError("missing_value must be a non-empty string")

    @property
    def subject(self) -> str:
        return self.relation.subject

    @property
    def predicate(self) -> str:
        return self.relation.predicate

    @property
    def object(self) -> str:
        return self.relation.object

    @property
    def origin(self) -> str:
        return self.relation.origin.value

    @property
    def evidence(self) -> str | None:
        return self.relation.evidence

    @property
    def confidence(self) -> float | None:
        return self.relation.confidence

    @property
    def source_id(self) -> str | None:
        return self.relation.source_id

    def to_dict(self) -> dict[str, object]:
        return {
            "missing_value": self.missing_value,
            "scenario_id": self.scenario_id,
            "dataset_id": self.relation.dataset_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "origin": self.origin,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "source_location": self.relation.source_location,
            "scope": self.scope.value,
            "relevance_score": self.relevance_score,
        }


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    scenario_id: str
    status: RecoveryStatus
    candidates: tuple[RecoveryCandidate, ...] = ()
    retrieval_scope: RetrievalScope = RetrievalScope.SCENARIO_LOCAL
    graph_context: tuple[RetrievedEvidence, ...] = ()
    scenario_context: str = ""
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "graph_context", tuple(self.graph_context))

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "retrieval_scope": self.retrieval_scope.value,
            "is_empty": self.is_empty,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "scenario_context": self.scenario_context,
            "graph_context": [
                {
                    "scenario_id": item.scenario_id,
                    "dataset_id": item.dataset_id,
                    "subject": item.relation.subject,
                    "predicate": item.relation.predicate,
                    "object": item.relation.object,
                    "origin": item.relation.origin.value,
                    "source_id": item.relation.source_id,
                    "scope": item.scope.value,
                }
                for item in self.graph_context
            ],
        }


class RecoverMissingLink:
    """Recover candidate links using an injected retrieval strategy."""

    def __init__(self, retriever: RetrievalStrategy) -> None:
        self._retriever = retriever

    def execute(self, request: RecoveryRequest) -> RecoveryResult:
        retrieval_request = RetrievalRequest(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            predicate=request.predicate.strip(),
            known_endpoint=request.known_endpoint.strip(),
            missing_endpoint=request.missing_endpoint.value,
            limit=request.limit,
        )
        retrieval_result = self._retriever.retrieve(retrieval_request)

        if not retrieval_result.scenario_exists:
            return RecoveryResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                status=RecoveryStatus.SCENARIO_NOT_FOUND,
                retrieval_scope=retrieval_result.scope,
            )

        candidates: list[RecoveryCandidate] = []
        seen: set[str] = set()
        for item in retrieval_result.evidence:
            relation = item.relation
            if relation.origin.value == "model_inferred":
                continue
            missing_value = (
                relation.subject
                if request.missing_endpoint == MissingEndpoint.SUBJECT
                else relation.object
            )
            normalized_value = missing_value.strip().casefold()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            candidates.append(
                RecoveryCandidate(
                    missing_value=missing_value,
                    relation=relation,
                    scenario_id=item.scenario_id,
                    scope=item.scope,
                    relevance_score=item.relevance_score,
                )
            )
            if len(candidates) >= request.limit:
                break

        return RecoveryResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            status=(
                RecoveryStatus.CANDIDATES_FOUND
                if candidates
                else RecoveryStatus.NO_CANDIDATES
            ),
            candidates=tuple(candidates),
            retrieval_scope=retrieval_result.scope,
            graph_context=retrieval_result.graph_context,
            scenario_context=retrieval_result.scenario_context,
        )
