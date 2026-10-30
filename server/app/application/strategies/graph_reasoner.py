"""Graph-based reasoning for structured HAZOP missing-link recovery."""

from __future__ import annotations

from collections import defaultdict, deque

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


class GraphReasoner:
    """Reason over an evidence graph without inventing unsupported links.

    This baseline uses directed graph structure to inspect candidate
    relationships. It proposes a relationship only when exactly one
    evidence-backed relation is supported by the graph.
    """

    _ELIGIBLE_ORIGINS = {
        RelationshipOrigin.SOURCE,
        RelationshipOrigin.EXPERT,
    }

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Return a supported relationship or abstain."""

        if not context.evidence_items:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
            )

        eligible = [
            item
            for item in context.evidence_items
            if item.origin in self._ELIGIBLE_ORIGINS
            and item.evidence.strip()
        ]

        if not eligible:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        adjacency: dict[str, set[str]] = defaultdict(set)

        for item in eligible:
            adjacency[item.subject.casefold()].add(
                item.object.casefold()
            )

        # Identify evidence-backed relations whose endpoints are connected
        # through another path in the graph.
        candidates = []

        for item in eligible:
            subject = item.subject.casefold()
            object_ = item.object.casefold()

            if self._has_alternate_path(
                adjacency,
                subject,
                object_,
            ):
                candidates.append(item)

        if not candidates:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        if len(candidates) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                supporting_evidence=tuple(
                    item.evidence for item in candidates
                ),
                abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
            )

        item = candidates[0]

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
    def _has_alternate_path(
        adjacency: dict[str, set[str]],
        start: str,
        target: str,
    ) -> bool:
        """Check whether another directed path connects two nodes."""

        queue: deque[str] = deque(adjacency.get(start, set()))
        visited: set[str] = {start}

        while queue:
            current = queue.popleft()

            if current == target:
                return True

            if current in visited:
                continue

            visited.add(current)
            queue.extend(adjacency.get(current, set()) - visited)

        return False

    @staticmethod
    def _abstain(
        scenario_id: str,
        reason: AbstentionReason,
    ) -> ReasoningResult:
        """Create a standardized abstention result."""

        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            abstention_reason=reason,
        )
