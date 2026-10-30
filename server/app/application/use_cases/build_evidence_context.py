"""Build provenance-aware evidence contexts for HAZOP reasoning."""

from __future__ import annotations

from dataclasses import dataclass

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

    @classmethod
    def from_candidate(
        cls,
        candidate: RecoveryCandidate,
    ) -> EvidenceItem:
        """Convert a recovered candidate into an evidence item."""
        relation = candidate.relation

        if relation.origin == RelationshipOrigin.SOURCE:
            note = "Relationship attributed to a source."
        elif relation.origin == RelationshipOrigin.EXPERT:
            note = "Relationship attributed to expert validation."
        else:
            note = (
                "Relationship inferred by a model; "
                "not established evidence."
            )

        return cls(
            subject=relation.subject,
            predicate=relation.predicate,
            object=relation.object,
            origin=relation.origin,
            evidence=relation.evidence,
            confidence=relation.confidence,
            source_id=relation.source_id,
            relevance_note=note,
        )

    def to_dict(self) -> dict[str, str | float | None]:
        """Return a JSON-friendly representation."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "origin": self.origin.value,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "relevance_note": self.relevance_note,
        }


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    """Evidence and the original query required for reasoning."""

    scenario_id: str
    predicate: str
    known_endpoint: str
    missing_endpoint: str
    status: RecoveryStatus
    evidence_items: tuple[EvidenceItem, ...]

    @property
    def is_empty(self) -> bool:
        """Whether the context contains any evidence."""
        return not self.evidence_items

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "scenario_id": self.scenario_id,
            "predicate": self.predicate,
            "known_endpoint": self.known_endpoint,
            "missing_endpoint": self.missing_endpoint,
            "status": self.status.value,
            "is_empty": self.is_empty,
            "evidence_items": [
                item.to_dict() for item in self.evidence_items
            ],
        }


class BuildEvidenceContext:
    """Construct reasoning context from missing-link recovery results."""

    def execute(
        self,
        recovery_result: RecoveryResult,
        request: RecoveryRequest,
    ) -> EvidenceContext:
        """Combine recovery evidence with the original query."""
        evidence_items = tuple(
            EvidenceItem.from_candidate(candidate)
            for candidate in recovery_result.candidates
        )

        if recovery_result.scenario_id != request.scenario_id:
            raise ValueError(
                "Recovery result scenario_id must match the request."
            )

        return EvidenceContext(
            scenario_id=request.scenario_id,
            predicate=request.predicate,
            known_endpoint=request.known_endpoint,
            missing_endpoint=request.missing_endpoint.value,
            status=recovery_result.status,
            evidence_items=evidence_items,
        )
