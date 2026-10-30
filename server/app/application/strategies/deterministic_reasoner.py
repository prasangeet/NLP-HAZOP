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

        dataset_id = getattr(context, "dataset_id", "default")
        if not context.evidence_items:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
                dataset_id,
            )

        missing_endpoint = context.missing_endpoint.casefold()
        if missing_endpoint not in {"subject", "object"}:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
            )

        predicate = context.predicate.strip().casefold()
        known_endpoint = context.known_endpoint.strip().casefold()

        eligible = [
            item
            for item in context.evidence_items
            if getattr(item, "dataset_id", "default") == dataset_id
            and item.origin
            in {
                RelationshipOrigin.SOURCE,
                RelationshipOrigin.RECONSTRUCTED,
                RelationshipOrigin.SYNTHETIC,
                RelationshipOrigin.EXPERT,
            }
            and item.evidence.strip()
            and (
                item.origin != RelationshipOrigin.EXPERT
                or getattr(item, "is_expert_verified", False)
            )
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
                dataset_id,
            )

        if len(eligible) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                dataset_id=getattr(context, "dataset_id", "default"),
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
            dataset_id=getattr(item, "dataset_id", dataset_id),
            is_expert_verified=getattr(item, "is_expert_verified", False),
        )

        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            dataset_id=dataset_id,
            proposed_relation=relation,
            supporting_evidence=(item.evidence,),
            confidence=item.confidence,
        )

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
        dataset_id: str = "default",
    ) -> ReasoningResult:
        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            abstention_reason=reason,
        )
