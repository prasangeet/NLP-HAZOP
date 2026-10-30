"""Build provenance-aware evidence contexts for HAZOP reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from app.application.strategies.retrieval.models import (
    RetrievalScope,
    RetrievedEvidence,
)
from app.application.use_cases.recover_missing_link import (
    RecoveryCandidate,
    RecoveryRequest,
    RecoveryResult,
    RecoveryStatus,
)
from app.domain.entities.hazop_record import RelationshipOrigin


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A relationship presented as evidence to a reasoning strategy."""

    subject: str
    predicate: str
    object: str
    origin: RelationshipOrigin
    evidence: str
    confidence: float | None
    source_id: str | None
    relevance_note: str
    scenario_id: str = ""
    scope: RetrievalScope = RetrievalScope.SCENARIO_LOCAL
    relevance_score: float | None = None
    dataset_id: str = "default"
    is_expert_verified: bool = False

    @classmethod
    def from_candidate(cls, candidate: RecoveryCandidate) -> EvidenceItem:
        relation = candidate.relation
        return cls(
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            origin=relation.origin,
            evidence=relation.evidence,
            confidence=relation.confidence,
            source_id=relation.source_id,
            relevance_note=cls._origin_note(
                relation.origin, relation.is_expert_verified
            ),
            scenario_id=candidate.scenario_id,
            scope=candidate.scope,
            relevance_score=candidate.relevance_score,
            dataset_id=relation.dataset_id,
            is_expert_verified=relation.is_expert_verified,
        )

    @classmethod
    def from_retrieved(cls, item: RetrievedEvidence) -> EvidenceItem:
        relation = item.relation
        return cls(
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            origin=relation.origin,
            evidence=relation.evidence,
            confidence=relation.confidence,
            source_id=relation.source_id,
            relevance_note=cls._origin_note(
                relation.origin, relation.is_expert_verified
            ),
            scenario_id=item.scenario_id,
            scope=item.scope,
            relevance_score=item.relevance_score,
            dataset_id=relation.dataset_id,
            is_expert_verified=relation.is_expert_verified,
        )

    @staticmethod
    def _origin_note(
        origin: RelationshipOrigin,
        is_expert_verified: bool = False,
    ) -> str:
        notes = {
            RelationshipOrigin.SOURCE: "Directly attributed to a source.",
            RelationshipOrigin.RECONSTRUCTED: (
                "Reconstructed from a source report; not a verbatim row."
            ),
            RelationshipOrigin.SYNTHETIC: (
                "Synthetic scenario; not source-verified evidence."
            ),
            RelationshipOrigin.EXPERT: (
                "This individual relationship has explicit expert validation."
                if is_expert_verified
                else (
                    "The record carries expert validation provenance, but "
                    "this individual relationship is not independently "
                    "expert-validated."
                )
            ),
            RelationshipOrigin.UNSPECIFIED: (
                "Origin is unspecified; this relationship must not be "
                "treated as source-verified or expert-validated evidence."
            ),
            RelationshipOrigin.MODEL_INFERRED: (
                "Model-inferred; not established evidence."
            ),
            RelationshipOrigin.RETRIEVAL_INFERRED: (
                "Vector-retrieval candidate derived from row-level "
                "co-occurrence; not a direct source-verified causal edge."
            ),
            RelationshipOrigin.GRAPH_INFERRED: (
                "Derived by graph composition; underlying path is shown."
            ),
        }
        return notes[origin]

    def to_dict(self) -> dict[str, str | float | bool | None]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "origin": self.origin.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "relevance_note": self.relevance_note,
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "is_expert_verified": self.is_expert_verified,
            "scope": self.scope.value,
            "relevance_score": self.relevance_score,
        }


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Evidence, local graph neighbourhood, and original query."""

    scenario_id: str
    predicate: str
    known_endpoint: str
    missing_endpoint: str
    status: RecoveryStatus
    evidence_items: tuple[EvidenceItem, ...]
    graph_context: tuple[EvidenceItem, ...] = ()
    scenario_context: str = ""
    dataset_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_items", tuple(self.evidence_items))
        object.__setattr__(self, "graph_context", tuple(self.graph_context))

    @property
    def is_empty(self) -> bool:
        return not self.evidence_items and not self.graph_context

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "predicate": self.predicate,
            "known_endpoint": self.known_endpoint,
            "missing_endpoint": self.missing_endpoint,
            "status": self.status.value,
            "is_empty": self.is_empty,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "graph_context": [item.to_dict() for item in self.graph_context],
            "scenario_context": self.scenario_context,
        }


class BuildEvidenceContext:
    """Construct reasoning context from missing-link recovery results."""

    def execute(
        self,
        recovery_result: RecoveryResult,
        request: RecoveryRequest,
    ) -> EvidenceContext:
        if (
            recovery_result.scenario_id != request.scenario_id
            or recovery_result.dataset_id != request.dataset_id
        ):
            raise ValueError(
                "Recovery result dataset_id and scenario_id must match the request."
            )

        evidence_items = tuple(
            EvidenceItem.from_candidate(candidate)
            for candidate in recovery_result.candidates
        )
        graph_context = tuple(
            EvidenceItem.from_retrieved(item)
            for item in recovery_result.graph_context
        )
        return EvidenceContext(
            scenario_id=request.scenario_id,
            predicate=request.predicate,
            known_endpoint=request.known_endpoint,
            missing_endpoint=request.missing_endpoint.value,
            status=recovery_result.status,
            evidence_items=evidence_items,
            graph_context=graph_context or evidence_items,
            scenario_context=recovery_result.scenario_context,
            dataset_id=request.dataset_id,
        )
