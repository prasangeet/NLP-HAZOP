"""Conservative evidence-based reasoning baseline."""

from __future__ import annotations

from app.domain.reasoning.models import EvidenceContextProtocol
from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    ReasoningResult,
    ReasoningStatus,
)


class DeterministicReasoner:
    """Propose a link only when one eligible candidate exists."""

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return a supported proposal or abstain."""

        if not context.evidence_items:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                abstention_reason=AbstentionReason.NO_EVIDENCE,
            )

        eligible = [
            item
            for item in context.evidence_items
            if item.origin
            in {
                RelationshipOrigin.SOURCE,
                RelationshipOrigin.EXPERT,
            }
            and item.evidence.strip()
        ]

        if not eligible:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                abstention_reason=AbstentionReason.INSUFFICIENT_SUPPORT,
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
