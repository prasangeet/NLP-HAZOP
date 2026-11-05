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
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import EntityType


# An excluded relation is identified by:
# (scenario_id, subject, predicate, object)
ExcludedRelation = tuple[str, str, str, str]


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
    excluded_relations: tuple[ExcludedRelation, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "scenario_id",
            "dataset_id",
            "predicate",
            "known_endpoint",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string"
                )

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

        normalized_exclusions: list[ExcludedRelation] = []

        for exclusion in self.excluded_relations:
            if (
                not isinstance(exclusion, (tuple, list))
                or len(exclusion) != 4
                or not all(
                    isinstance(value, str) and value.strip()
                    for value in exclusion
                )
            ):
                raise ValueError(
                    "Each excluded relation must contain "
                    "scenario_id, subject, predicate, and object."
                )

            normalized_exclusions.append(
                tuple(value.strip() for value in exclusion)
            )

        object.__setattr__(
            self,
            "excluded_relations",
            tuple(normalized_exclusions),
        )


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    """A recovered candidate preserving its original relation and provenance."""

    missing_value: str
    relation: CausalRelation
    scenario_id: str = ""
    scope: RetrievalScope = RetrievalScope.SCENARIO_LOCAL
    relevance_score: float | None = None
    process_context: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.missing_value, str)
            or not self.missing_value.strip()
        ):
            raise ValueError("missing_value must be a non-empty string")

        object.__setattr__(
            self,
            "missing_value",
            self.missing_value.strip(),
        )

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
            "process_context": self.process_context,
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
    scenario_entities: tuple[tuple[str, EntityType], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "graph_context", tuple(self.graph_context))
        object.__setattr__(
            self,
            "scenario_entities",
            tuple(self.scenario_entities),
        )

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
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "scenario_context": self.scenario_context,
            "scenario_entities": [
                {
                    "value": value,
                    "type": entity_type.value,
                }
                for value, entity_type in self.scenario_entities
            ],
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
        """Retrieve evidence and exclude explicitly held-out relations."""

        retrieval_request = RetrievalRequest(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            predicate=request.predicate,
            known_endpoint=request.known_endpoint,
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

        # Apply the same exclusion policy to both retrieval evidence and
        # graph context. This prevents the held-out relation from reaching
        # candidate generation, GraphReasoner, or LLMReasoner.
        filtered_evidence = tuple(
            item
            for item in retrieval_result.evidence
            if not self._is_excluded(item, request)
        )

        filtered_graph_context = tuple(
            item
            for item in retrieval_result.graph_context
            if not self._is_excluded(item, request)
        )

        candidates = self._build_candidates(
            filtered_evidence,
            request,
        )

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
            graph_context=filtered_graph_context,
            scenario_context=retrieval_result.scenario_context,
            scenario_entities=retrieval_result.scenario_entities,
        )

    @staticmethod
    def _is_excluded(
        item: RetrievedEvidence,
        request: RecoveryRequest,
    ) -> bool:
        """Check whether an evidence relation is explicitly held out."""

        relation = item.relation

        actual = (
            item.scenario_id.strip().casefold(),
            relation.subject.strip().casefold(),
            relation.predicate.strip().casefold(),
            relation.object.strip().casefold(),
        )

        return any(
            actual
            == tuple(value.casefold() for value in excluded)
            for excluded in request.excluded_relations
        )

    @classmethod
    def _build_candidates(
        cls,
        evidence: tuple[RetrievedEvidence, ...],
        request: RecoveryRequest,
    ) -> list[RecoveryCandidate]:
        """Build unique, request-compatible candidates from retrieved evidence."""

        candidates: list[RecoveryCandidate] = []
        seen: set[str] = set()

        for item in evidence:
            relation = item.relation

            if not cls._is_eligible(item, request):
                continue

            missing_value = cls._get_missing_value(
                relation,
                request.missing_endpoint,
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
                    process_context=item.process_context,
                )
            )

            if len(candidates) >= request.limit:
                break

        return candidates

    @staticmethod
    def _is_eligible(
        item: RetrievedEvidence,
        request: RecoveryRequest,
    ) -> bool:
        """Validate dataset, predicate, provenance, and endpoint direction."""

        relation = item.relation

        # Model-generated hypotheses are not retrieval evidence.
        if relation.origin == RelationshipOrigin.MODEL_INFERRED:
            return False

        if item.dataset_id != request.dataset_id:
            return False

        if relation.predicate.strip().casefold() != request.predicate.casefold():
            return False

        if not relation.subject.strip() or not relation.object.strip():
            return False

        if request.missing_endpoint == MissingEndpoint.OBJECT:
            return (
                relation.subject.strip().casefold()
                == request.known_endpoint.casefold()
            )

        return (
            relation.object.strip().casefold()
            == request.known_endpoint.casefold()
        )

    @staticmethod
    def _get_missing_value(
        relation: CausalRelation,
        missing_endpoint: MissingEndpoint,
    ) -> str:
        """Extract the endpoint requested for recovery."""

        if missing_endpoint == MissingEndpoint.SUBJECT:
            return relation.subject

        return relation.object
