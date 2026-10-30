"""Conservative evidence-based reasoning baseline."""

from __future__ import annotations

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    EvidenceContextProtocol,
    ReasoningResult,
    ReasoningStatus,
)


class DeterministicReasoner:
    """Propose a query-aligned link only when one candidate exists."""

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return a supported proposal or abstain."""

        if not context.evidence_items:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
            )

        missing_endpoint = context.missing_endpoint.casefold()
        if missing_endpoint not in {"subject", "object"}:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        predicate = context.predicate.strip().casefold()
        known_endpoint = context.known_endpoint.strip().casefold()

        eligible = [
            item
            for item in context.evidence_items
            if item.origin
            in {
                RelationshipOrigin.SOURCE,
                RelationshipOrigin.EXPERT,
            }
            and item.evidence.strip()
            and item.predicate.strip().casefold() == predicate
            and (
                item.subject.strip().casefold() == known_endpoint
                if missing_endpoint == "object"
                else item.object.strip().casefold() == known_endpoint
            )
        ]

        if not eligible:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        if len(eligible) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
                supporting_evidence=tuple(
                    item.evidence for item in eligible
                ),
            )

        item = eligible[0]
        relation = CausalRelation(
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            origin=item.origin,
            evidence=item.evidence,
            confidence=item.confidence,
            source_id=item.source_id,
        )

        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            proposed_relation=relation,
            supporting_evidence=(item.evidence,),
            confidence=item.confidence,
        )

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
    ) -> ReasoningResult:
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            abstention_reason=reason,
        )
