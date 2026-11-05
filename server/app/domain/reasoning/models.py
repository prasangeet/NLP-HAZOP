
"""Provider-independent models for missing-link reasoning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)


@runtime_checkable
class EvidenceItemProtocol(Protocol):
    """Minimum read-only evidence interface required by a reasoning strategy."""

    @property
    def subject(self) -> str: ...

    @property
    def predicate(self) -> str: ...

    @property
    def object(self) -> str: ...

    @property
    def evidence(self) -> str: ...

    @property
    def origin(self) -> RelationshipOrigin: ...

    @property
    def confidence(self) -> float | None: ...

    @property
    def source_id(self) -> str | None: ...

    @property
    def dataset_id(self) -> str: ...

    @property
    def is_expert_verified(self) -> bool: ...


@runtime_checkable
class EvidenceContextProtocol(Protocol):
    """Read-only evidence context required by missing-link reasoning."""

    @property
    def scenario_id(self) -> str: ...

    @property
    def predicate(self) -> str: ...

    @property
    def known_endpoint(self) -> str: ...

    @property
    def missing_endpoint(self) -> str: ...

    @property
    def evidence_items(self) -> tuple[EvidenceItemProtocol, ...]: ...

    @property
    def graph_context(self) -> tuple[EvidenceItemProtocol, ...]: ...

    @property
    def scenario_context(self) -> str: ...


class ReasoningStatus(StrEnum):
    """Outcome of a reasoning attempt."""

    PROPOSED = "proposed"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    """Why a reasoning strategy declined to propose a relationship."""

    NO_EVIDENCE = "no_evidence"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    INSUFFICIENT_SUPPORT = "insufficient_support"


class DiagnosticStatus(StrEnum):
    """Outcome of a diagnostic generation attempt."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DiagnosticIssue:
    """A specific issue identified during recovery diagnosis."""

    component: str
    issue: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("Diagnostic issue component must not be empty.")

        if not isinstance(self.issue, str) or not self.issue.strip():
            raise ValueError("Diagnostic issue description must not be empty.")

        object.__setattr__(self, "component", self.component.strip())
        object.__setattr__(self, "issue", self.issue.strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "issue": self.issue,
        }


@dataclass(frozen=True, slots=True)
class ReasoningDiagnostic:
    """Provider-independent diagnostic generated for a recovery attempt."""

    generated_by: str
    status: DiagnosticStatus
    message: str
    identified_issues: tuple[DiagnosticIssue, ...] = ()
    recommended_action: str = ""
    error_category: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generated_by, str) or not self.generated_by.strip():
            raise ValueError("Diagnostic generated_by must not be empty.")

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Diagnostic message must not be empty.")

        if not isinstance(self.recommended_action, str):
            raise ValueError("Diagnostic recommended_action must be a string.")

        normalized_issues: list[DiagnosticIssue] = []

        for issue in self.identified_issues:
            if isinstance(issue, DiagnosticIssue):
                normalized_issues.append(issue)
            elif isinstance(issue, dict):
                normalized_issues.append(DiagnosticIssue(**issue))
            else:
                raise TypeError(
                    "identified_issues must contain DiagnosticIssue objects."
                )

        if self.status == DiagnosticStatus.COMPLETED and self.error_category:
            raise ValueError(
                "A completed diagnostic cannot have an error category."
            )

        if self.status == DiagnosticStatus.FAILED and not self.error_category:
            raise ValueError(
                "A failed diagnostic must include an error category."
            )

        object.__setattr__(self, "generated_by", self.generated_by.strip())
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(
            self,
            "recommended_action",
            self.recommended_action.strip(),
        )
        object.__setattr__(
            self,
            "identified_issues",
            tuple(normalized_issues),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the diagnostic into a JSON-friendly dictionary."""

        return {
            "generated_by": self.generated_by,
            "status": self.status.value,
            "message": self.message,
            "identified_issues": [
                issue.to_dict() for issue in self.identified_issues
            ],
            "recommended_action": self.recommended_action,
            "error_category": self.error_category,
        }


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Structured output from a missing-link reasoning strategy."""

    status: ReasoningStatus
    scenario_id: str
    proposed_relation: CausalRelation | None = None
    supporting_evidence: tuple[str, ...] = ()
    confidence: float | None = None
    abstention_reason: AbstentionReason | None = None
    dataset_id: str = "default"
    diagnostic: ReasoningDiagnostic | None = None

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

        object.__setattr__(
            self,
            "scenario_id",
            self.scenario_id.strip(),
        )

        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty.")

        dataset_id = self.dataset_id.strip()

        if self.proposed_relation is not None:
            relation = self.proposed_relation

            if relation.dataset_id == "default" and dataset_id != "default":
                relation = replace(
                    relation,
                    dataset_id=dataset_id,
                )
                object.__setattr__(
                    self,
                    "proposed_relation",
                    relation,
                )

            elif dataset_id == "default" and relation.dataset_id != "default":
                dataset_id = relation.dataset_id

            elif relation.dataset_id != dataset_id:
                raise ValueError(
                    "proposed_relation.dataset_id must match result dataset_id."
                )

        object.__setattr__(self, "dataset_id", dataset_id)

        object.__setattr__(
            self,
            "supporting_evidence",
            tuple(
                evidence.strip()
                for evidence in self.supporting_evidence
                if isinstance(evidence, str) and evidence.strip()
            ),
        )

        if self.diagnostic is not None and not isinstance(
            self.diagnostic,
            ReasoningDiagnostic,
        ):
            raise TypeError(
                "diagnostic must be a ReasoningDiagnostic instance or None."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the result into a JSON-friendly dictionary."""

        relation = self.proposed_relation

        return {
            "status": self.status.value,
            "scenario_id": self.scenario_id,
            "dataset_id": self.dataset_id,
            "proposed_relation": (
                {
                    "subject": relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object,
                    "origin": relation.origin.value,
                    "evidence": relation.evidence,
                    "confidence": relation.confidence,
                    "source_id": relation.source_id,
                    "dataset_id": relation.dataset_id,
                    "is_expert_verified": relation.is_expert_verified,
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
            "diagnostic": (
                self.diagnostic.to_dict()
                if self.diagnostic is not None
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
