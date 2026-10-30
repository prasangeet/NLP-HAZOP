"""Conservative graph-structural baseline for HAZOP link recovery."""

from __future__ import annotations

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    EvidenceContextProtocol,
    EvidenceItemProtocol,
    ReasoningResult,
    ReasoningStatus,
)


class GraphReasoner:
    """Recover a missing endpoint from query-aligned graph evidence.

    The current retriever supplies relationships that match the requested
    predicate and known endpoint. This strategy validates that alignment,
    rejects model-inferred or undocumented evidence, and proposes only when
    exactly one distinct missing endpoint remains.

    This is a conservative one-hop graph baseline. It does not claim multi-hop
    path reasoning because the current evidence context does not contain the
    full set of neighboring graph edges.
    """

    _ELIGIBLE_ORIGINS = frozenset(
        {
            RelationshipOrigin.SOURCE,
            RelationshipOrigin.EXPERT,
        }
    )

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return one uniquely supported relationship or abstain."""

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

        eligible: list[EvidenceItemProtocol] = []
        for item in context.evidence_items:
            if item.origin not in self._ELIGIBLE_ORIGINS:
                continue
            if not item.evidence.strip():
                continue
            if item.predicate.strip().casefold() != predicate:
                continue

            if missing_endpoint == "object":
                aligned = item.subject.strip().casefold() == known_endpoint
            else:
                aligned = item.object.strip().casefold() == known_endpoint

            if aligned:
                eligible.append(item)

        if not eligible:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        # Several records supporting one endpoint are corroboration, not
        # competing endpoint candidates.
        grouped: dict[str, list[EvidenceItemProtocol]] = {}
        for item in eligible:
            missing_value = (
                item.object if missing_endpoint == "object" else item.subject
            )
            grouped.setdefault(missing_value.strip().casefold(), []).append(item)

        if len(grouped) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                supporting_evidence=tuple(
                    item.evidence for item in eligible
                ),
                abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
            )

        supporting_items = next(iter(grouped.values()))
        representative = supporting_items[0]

        # Preserve the exact retrieved triple and its original provenance.
        relation = CausalRelation(
            subject=representative.subject,
            predicate=representative.predicate,
            object=representative.object,
            origin=representative.origin,
            evidence=representative.evidence,
            confidence=representative.confidence,
            source_id=representative.source_id,
        )

        confidences = [
            item.confidence
            for item in supporting_items
            if item.confidence is not None
        ]
        confidence = max(confidences) if confidences else None

        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            proposed_relation=relation,
            supporting_evidence=tuple(
                dict.fromkeys(item.evidence for item in supporting_items)
            ),
            confidence=confidence,
        )

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
    ) -> ReasoningResult:
        """Build a standardized abstention result."""

        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            abstention_reason=reason,
        )
