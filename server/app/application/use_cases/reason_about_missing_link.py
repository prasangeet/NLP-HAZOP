"""Orchestrate missing-link recovery and evidence-based reasoning."""

from __future__ import annotations

from app.application.use_cases.build_evidence_context import (
    BuildEvidenceContext,
)
from app.application.use_cases.recover_missing_link import (
    RecoverMissingLink,
    RecoveryRequest,
    RecoveryStatus,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStrategy,
)


class ReasonAboutMissingLink:
    """Coordinate retrieval, context construction, and reasoning."""

    def __init__(
        self,
        recovery_use_case: RecoverMissingLink,
        evidence_context_builder: BuildEvidenceContext,
        reasoning_strategy: ReasoningStrategy,
    ) -> None:
        self._recovery_use_case = recovery_use_case
        self._evidence_context_builder = evidence_context_builder
        self._reasoning_strategy = reasoning_strategy

    def execute(
        self,
        request: RecoveryRequest,
    ) -> ReasoningResult:
        """Recover evidence and reason about the missing relationship."""
        recovery_result = self._recovery_use_case.execute(request)

        if recovery_result.status == RecoveryStatus.SCENARIO_NOT_FOUND:
            return ReasoningResult(
                status="abstained",
                scenario_id=request.scenario_id,
                abstention_reason=AbstentionReason.NO_EVIDENCE,
            )

        evidence_context = self._evidence_context_builder.execute(
            recovery_result,
            request,
        )

        return self._reasoning_strategy.reason(evidence_context)
