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
            relevance_note=cls._origin_note(relation.origin),
            scenario_id=candidate.scenario_id,
            scope=candidate.scope,
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
            relevance_note=cls._origin_note(relation.origin),
            scenario_id=item.scenario_id,
            scope=item.scope,
        )

    @staticmethod
    def _origin_note(origin: RelationshipOrigin) -> str:
        notes = {
            RelationshipOrigin.SOURCE: "Directly attributed to a source.",
            RelationshipOrigin.RECONSTRUCTED: (
                "Reconstructed from a source report; not a verbatim row."
            ),
            RelationshipOrigin.SYNTHETIC: (
                "Synthetic scenario; not source-verified evidence."
            ),
            RelationshipOrigin.EXPERT: "Attributed to expert validation.",
            RelationshipOrigin.MODEL_INFERRED: (
                "Model-inferred; not established evidence."
            ),
            RelationshipOrigin.GRAPH_INFERRED: (
                "Derived by graph composition; underlying path is shown."
            ),
        }
        return notes[origin]

    def to_dict(self) -> dict[str, str | float | None]:
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
            "scope": self.scope.value,
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_items", tuple(self.evidence_items))
        object.__setattr__(self, "graph_context", tuple(self.graph_context))

    @property
    def is_empty(self) -> bool:
        return not self.evidence_items and not self.graph_context

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
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
        if recovery_result.scenario_id != request.scenario_id:
            raise ValueError(
                "Recovery result scenario_id must match the request."
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
        )
