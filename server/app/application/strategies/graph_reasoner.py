"""Graph reasoning over typed, provenance-aware HAZOP relationships."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class _Path:
    """A directed path through graph evidence."""

    nodes: tuple[str, ...]
    edges: tuple[EvidenceItemProtocol, ...]

    @property
    def endpoint(self) -> str:
        return self.nodes[-1]


class GraphReasoner:
    """Recover a candidate using direct edges or typed multi-hop paths.

    Direct evidence that matches the requested predicate is preferred.
    When no direct edge is available, a bounded path search can compose
    explicitly typed causal edges. A path-derived result is marked
    GRAPH_INFERRED and contains the complete path in its evidence text.
    Such a result is an indirect reachability hypothesis, not a claim that
    the first node is the immediate cause of the final node.
    """

    _DIRECT_ORIGINS = frozenset(
        {
            RelationshipOrigin.SOURCE,
            RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.SYNTHETIC,
            RelationshipOrigin.EXPERT,
        }
    )
    _CAUSAL_PREDICATES = frozenset(
        {
            "causes",
            "causes_deviation",
            "leads_to",
            "leads_to_consequence",
            "results_in",
        }
    )

    def __init__(self, max_hops: int = 4) -> None:
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise ValueError("max_hops must be an integer")
        if max_hops < 2:
            raise ValueError("max_hops must be at least 2")
        self._max_hops = max_hops

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Propose a unique direct candidate or a unique graph path endpoint."""

        if not context.evidence_items and not context.graph_context:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
            )

        direction = context.missing_endpoint.strip().casefold()
        if direction not in {"subject", "object"}:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        direct = self._direct_candidates(context, direction)
        if direct:
            return self._resolve_direct(context.scenario_id, direct, direction)

        paths = self._find_paths(context, direction)
        if not paths:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
            )

        by_endpoint: dict[str, list[_Path]] = defaultdict(list)
        for path in paths:
            by_endpoint[path.endpoint.strip().casefold()].append(path)

        if len(by_endpoint) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                supporting_evidence=tuple(
                    self._format_path(path)
                    for group in by_endpoint.values()
                    for path in group
                ),
                abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
            )

        selected_paths = next(iter(by_endpoint.values()))
        selected = min(selected_paths, key=lambda path: len(path.edges))
        known = context.known_endpoint.strip()
        endpoint = selected.endpoint
        subject, object_ = (
            (endpoint, known)
            if direction == "subject"
            else (known, endpoint)
        )
        path_evidence = self._format_path(selected)
        path_source_ids = [
            edge.source_id for edge in selected.edges if edge.source_id
        ]
        relation = CausalRelation(
            subject=subject,
            predicate=context.predicate,
            object=object_,
            origin=RelationshipOrigin.GRAPH_INFERRED,
            evidence=(
                "Indirect causal reachability inferred by typed graph "
                f"composition. {path_evidence} This is not evidence that "
                "the endpoints are an immediate/direct cause-effect pair."
            ),
            confidence=None,
            source_id=(
                "graph-path:" + "->".join(path_source_ids)
                if path_source_ids
                else None
            ),
        )
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            proposed_relation=relation,
            supporting_evidence=tuple(
                self._format_path(path) for path in selected_paths
            ),
            confidence=None,
        )

    def _direct_candidates(
        self,
        context: EvidenceContextProtocol,
        direction: str,
    ) -> list[EvidenceItemProtocol]:
        predicate = context.predicate.strip().casefold()
        known = context.known_endpoint.strip().casefold()
        unique: dict[
            tuple[str, str, str, str, str | None],
            EvidenceItemProtocol,
        ] = {}

        for item in context.evidence_items:
            if item.origin not in self._DIRECT_ORIGINS:
                continue
            if not item.evidence.strip():
                continue
            if item.predicate.strip().casefold() != predicate:
                continue

            endpoint_matches = (
                item.object.strip().casefold() == known
                if direction == "subject"
                else item.subject.strip().casefold() == known
            )
            if not endpoint_matches:
                continue

            key = (
                item.subject.strip().casefold(),
                item.predicate.strip().casefold(),
                item.object.strip().casefold(),
                item.evidence.strip(),
                item.source_id,
            )
            unique.setdefault(key, item)

        return list(unique.values())

    def _resolve_direct(
        self,
        scenario_id: str,
        candidates: list[EvidenceItemProtocol],
        direction: str,
    ) -> ReasoningResult:
        endpoint_values = {
            (
                item.subject if direction == "subject" else item.object
            ).strip().casefold()
            for item in candidates
        }
        if len(endpoint_values) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=scenario_id,
                supporting_evidence=tuple(
                    item.evidence for item in candidates
                ),
                abstention_reason=AbstentionReason.AMBIGUOUS_EVIDENCE,
            )

        selected = candidates[0]
        relation = CausalRelation(
            subject=selected.subject,
            predicate=selected.predicate,
            object=selected.object,
            origin=selected.origin,
            evidence=selected.evidence,
            confidence=selected.confidence,
            source_id=selected.source_id,
        )
        confidences = [
            item.confidence
            for item in candidates
            if item.confidence is not None
        ]
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=scenario_id,
            proposed_relation=relation,
            supporting_evidence=tuple(
                dict.fromkeys(item.evidence for item in candidates)
            ),
            confidence=max(confidences) if confidences else None,
        )

    def _find_paths(
        self,
        context: EvidenceContextProtocol,
        direction: str,
    ) -> list[_Path]:
        known = context.known_endpoint.strip().casefold()
        adjacency: dict[str, list[tuple[str, EvidenceItemProtocol]]] = defaultdict(list)

        for edge in context.graph_context:
            if edge.origin not in self._DIRECT_ORIGINS:
                continue
            if not edge.evidence.strip():
                continue
            if edge.predicate.strip().casefold() not in self._CAUSAL_PREDICATES:
                continue

            subject = edge.subject.strip().casefold()
            object_ = edge.object.strip().casefold()
            if direction == "object":
                adjacency[subject].append((object_, edge))
            else:
                adjacency[object_].append((subject, edge))

        queue: deque[_Path] = deque([_Path((known,), ())])
        candidates: list[_Path] = []

        while queue:
            path = queue.popleft()
            if len(path.edges) >= self._max_hops:
                continue

            current = path.nodes[-1]
            for neighbour, edge in adjacency.get(current, []):
                if neighbour in path.nodes:
                    continue

                next_path = _Path(
                    nodes=(*path.nodes, neighbour),
                    edges=(*path.edges, edge),
                )

                if len(next_path.edges) >= 2:
                    # Require a cause edge followed by at least one
                    # propagation edge; this avoids treating arbitrary
                    # associations as transitive causal evidence.
                    predicates = [
                        item.predicate.strip().casefold()
                        for item in next_path.edges
                    ]
                    if direction == "object":
                        has_initiating_edge = predicates[0] in {
                            "causes",
                            "causes_deviation",
                        }
                        propagation_predicates = predicates[1:]
                        terminal_edge = predicates[-1] in {
                            "leads_to_consequence",
                            "results_in",
                        }
                    else:
                        # Traversal is reversed for a missing subject:
                        # consequence -> deviation -> cause.
                        has_initiating_edge = predicates[-1] in {
                            "causes",
                            "causes_deviation",
                        }
                        propagation_predicates = predicates[:-1]
                        terminal_edge = predicates[0] in {
                            "leads_to_consequence",
                            "results_in",
                        }

                    has_propagation_edge = any(
                        predicate in {
                            "leads_to",
                            "leads_to_consequence",
                            "results_in",
                        }
                        for predicate in propagation_predicates
                    )
                    dead_end = not adjacency.get(next_path.endpoint)
                    if (
                        has_initiating_edge
                        and has_propagation_edge
                        and (terminal_edge or dead_end)
                    ):
                        candidates.append(next_path)

                queue.append(next_path)

        return candidates

    @staticmethod
    def _format_path(path: _Path) -> str:
        pieces = [path.nodes[0]]
        for edge, next_node in zip(path.edges, path.nodes[1:]):
            provenance = edge.origin.value
            scenario_id = getattr(edge, "scenario_id", "")
            if scenario_id:
                provenance += f", scenario_id={scenario_id}"
            if edge.source_id:
                provenance += f", source_id={edge.source_id}"
            pieces.append(
                f"-[{edge.predicate} | {provenance}]-> {next_node}"
            )
        return " ".join(pieces)

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
