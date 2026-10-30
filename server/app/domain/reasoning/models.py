"""Provider-independent models for missing-link reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)


@runtime_checkable
class EvidenceItemProtocol(Protocol):
    """Minimum evidence interface required by a reasoning strategy."""

    subject: str
    predicate: str
    object: str
    evidence: str
    origin: RelationshipOrigin
    confidence: float | None
    source_id: str | None


@runtime_checkable
class EvidenceContextProtocol(Protocol):
    """Evidence context and query required for missing-link reasoning."""

    scenario_id: str
    predicate: str
    known_endpoint: str
    missing_endpoint: str
    evidence_items: tuple[EvidenceItemProtocol, ...]


class ReasoningStatus(StrEnum):
    """Outcome of a reasoning attempt."""

    PROPOSED = "proposed"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    """Why a reasoning strategy declined to propose a relationship."""

    NO_EVIDENCE = "no_evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    INSUFFICIENT_SUPPORT = "insufficient_support"


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Structured output from a missing-link reasoning strategy."""

    status: ReasoningStatus
    scenario_id: str
    proposed_relation: CausalRelation | None = None
    supporting_evidence: tuple[str, ...] = ()
    confidence: float | None = None
    abstention_reason: AbstentionReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty.")

        if self.confidence is not None and not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0.0 and 1.0.")

        if self.status == ReasoningStatus.PROPOSED:
            if self.proposed_relation is None:
                raise ValueError(
                    "A proposed result must include a relationship."
                )
            if self.abstention_reason is not None:
                raise ValueError(
                    "A proposed result cannot have an abstention reason."
                )
        elif self.status == ReasoningStatus.ABSTAINED:
            if self.proposed_relation is not None:
                raise ValueError(
                    "An abstained result cannot include a proposed relation."
                )
            if self.abstention_reason is None:
                raise ValueError(
                    "An abstained result must include a reason."
                )
        else:
            raise ValueError("Unsupported reasoning status.")

        object.__setattr__(self, "scenario_id", self.scenario_id.strip())
        object.__setattr__(
            self,
            "supporting_evidence",
            tuple(
                evidence.strip()
                for evidence in self.supporting_evidence
                if isinstance(evidence, str) and evidence.strip()
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the result into a JSON-friendly dictionary."""

        relation = self.proposed_relation
        return {
            "status": self.status.value,
            "scenario_id": self.scenario_id,
            "proposed_relation": (
                {
                    "subject": relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "confidence": relation.confidence,
                    "source_id": relation.source_id,
                }
                if relation is not None
                else None
            ),
            "supporting_evidence": list(self.supporting_evidence),
            "confidence": self.confidence,
            "abstention_reason": (
                self.abstention_reason.value
                if self.abstention_reason is not None
                else None
            ),
        }


@runtime_checkable
class ReasoningStrategy(Protocol):
    """Contract implemented by deterministic and LLM reasoners."""

    def reason(
        self,
        context: EvidenceContextProtocol,
    ) -> ReasoningResult:
        """Propose a missing relationship or abstain."""
        ...
