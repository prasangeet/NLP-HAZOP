"""Graph reasoning over typed, provenance-aware HAZOP relationships."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from app.domain.ontology.schema import (
    EntityType,
    RELATION_SCHEMA,
    validate_relation_types,
)
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

    nodes: tuple[tuple[EntityType, str], ...]
    display_nodes: tuple[str, ...]
    edges: tuple[EvidenceItemProtocol, ...]

    @property
    def endpoint(self) -> tuple[EntityType, str]:
        """Normalized terminal node for matching and deduplication."""
        return self.nodes[-1]

    @property
    def display_endpoint(self) -> str:
        """Original-case terminal node for the returned relationship."""
        return self.display_nodes[-1]


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
            RelationshipOrigin.LLM_EXTRACTED,
        }
    )
    _CAUSAL_PREDICATES = frozenset(
        {
            "causes_deviation",
            "leads_to",
            "leads_to_consequence",
        }
    )

    def __init__(self, max_hops: int = 4) -> None:
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise TypeError("max_hops must be an integer")
        if max_hops < 2:
            raise ValueError("max_hops must be at least 2")
        self._max_hops = max_hops

    def reason(self, context: EvidenceContextProtocol) -> ReasoningResult:
        """Propose a unique direct candidate or a unique graph path endpoint."""
        dataset_id = getattr(context, "dataset_id", "default")

        if not context.evidence_items and not context.graph_context:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.NO_EVIDENCE,
                dataset_id,
            )

        direction = context.missing_endpoint.strip().casefold()
        if direction not in {"subject", "object"}:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
            )

        direct = self._direct_candidates(context, direction)
        if direct:
            return self._resolve_direct(
                context.scenario_id, direct, direction, dataset_id
            )

        paths = self._find_paths(context, direction)
        if not paths:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
            )

        by_endpoint: dict[str, list[_Path]] = defaultdict(list)
        for path in paths:
            by_endpoint[path.endpoint].append(path)

        if len(by_endpoint) > 1:
            return ReasoningResult(
                status=ReasoningStatus.ABSTAINED,
                scenario_id=context.scenario_id,
                dataset_id=dataset_id,
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
        endpoint = selected.display_endpoint
        known_type = selected.nodes[0][0]
        endpoint_type = selected.nodes[-1][0]
        subject, object_ = (
            (endpoint, known)
            if direction == "subject"
            else (known, endpoint)
        )
        subject_type, object_type = (
            (endpoint_type, known_type)
            if direction == "subject"
            else (known_type, endpoint_type)
        )
        requested_spec = RELATION_SCHEMA.get(context.predicate.strip().casefold())
        if (
            requested_spec is None
            or subject_type not in requested_spec.subject_types
            or object_type not in requested_spec.object_types
        ):
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
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
                "Row-association candidate inferred by typed graph "
                f"composition. {path_evidence} The endpoints are co-listed "
                "through a reconstructed relationship path; this does not "
                "establish an independently validated causal assertion."
                if context.predicate.strip().casefold()
                == "co_listed_with_consequence"
                else
                "Indirect causal reachability inferred by typed graph "
                f"composition. {path_evidence} This is not evidence that "
                "the endpoints are an immediate/direct cause-effect pair."
            ),
            confidence=None,
            source_id=(
                f"graph-path:{dataset_id}:" + "->".join(path_source_ids)
                if path_source_ids
                else None
            ),
            dataset_id=dataset_id,
            subject_type=subject_type,
            object_type=object_type,
        )
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            dataset_id=dataset_id,
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

        dataset_id = getattr(context, "dataset_id", "default")
        for item in context.evidence_items:
            if getattr(item, "dataset_id", "default") != dataset_id:
                continue
            if item.origin not in self._DIRECT_ORIGINS:
                continue
            if (
                item.origin == RelationshipOrigin.EXPERT
                and not getattr(item, "is_expert_verified", False)
            ):
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
        dataset_id: str,
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
                dataset_id=dataset_id,
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
            dataset_id=getattr(selected, "dataset_id", dataset_id),
            is_expert_verified=getattr(selected, "is_expert_verified", False),
            subject_type=getattr(selected, "subject_type", EntityType.CONCEPT),
            object_type=getattr(selected, "object_type", EntityType.CONCEPT),
            source_location=getattr(selected, "source_location", None),
        )
        confidences = [
            item.confidence
            for item in candidates
            if item.confidence is not None
        ]
        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
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
        NodeKey = tuple[EntityType, str]
        adjacency: dict[
            NodeKey,
            list[tuple[NodeKey, EvidenceItemProtocol, str]],
        ] = defaultdict(list)
        start_nodes: list[NodeKey] = []

        dataset_id = getattr(context, "dataset_id", "default")
        for edge in context.graph_context:
            if getattr(edge, "dataset_id", "default") != dataset_id:
                continue
            if edge.origin not in self._DIRECT_ORIGINS:
                continue
            if (
                edge.origin == RelationshipOrigin.EXPERT
                and not getattr(edge, "is_expert_verified", False)
            ):
                continue
            if not edge.evidence.strip():
                continue
            if edge.predicate.strip().casefold() not in self._CAUSAL_PREDICATES:
                continue
            subject_type = EntityType(getattr(edge, "subject_type", EntityType.CONCEPT))
            object_type = EntityType(getattr(edge, "object_type", EntityType.CONCEPT))
            if validate_relation_types(edge.predicate, subject_type, object_type):
                continue

            subject: NodeKey = (subject_type, edge.subject.strip().casefold())
            object_: NodeKey = (object_type, edge.object.strip().casefold())
            if direction == "object":
                adjacency[subject].append((object_, edge, edge.object))
                if subject[1] == known:
                    start_nodes.append(subject)
            else:
                adjacency[object_].append((subject, edge, edge.subject))
                if object_[1] == known:
                    start_nodes.append(object_)

        queue: deque[_Path] = deque(
            _Path(
                nodes=(node,),
                display_nodes=(context.known_endpoint.strip(),),
                edges=(),
            )
            for node in dict.fromkeys(start_nodes)
        )
        candidates: list[_Path] = []

        while queue:
            path = queue.popleft()
            if len(path.edges) >= self._max_hops:
                continue

            current = path.nodes[-1]
            for neighbour, edge, display_value in adjacency.get(current, []):
                if neighbour in path.nodes:
                    continue

                next_path = _Path(
                    nodes=(*path.nodes, neighbour),
                    display_nodes=(*path.display_nodes, display_value),
                    edges=(*path.edges, edge),
                )

                if len(next_path.edges) >= 2:
                    # Compose only explicitly typed propagation paths.
                    # T-301 rows without an intermediate-effect column encode
                    # Cause -> Deviation -> Consequence (2 edges). Datasets
                    # with an explicit effect encode the 3-edge variant.
                    # These paths support a row-association candidate; they
                    # do not create a direct causal edge between endpoints.
                    predicates = [
                        item.predicate.strip().casefold()
                        for item in next_path.edges
                    ]
                    association_query = (
                        context.predicate.strip().casefold()
                        == "co_listed_with_consequence"
                    )
                    if association_query and direction == "object":
                        valid_chain = predicates in (
                            [
                                "causes_deviation",
                                "leads_to_consequence",
                            ],
                            [
                                "causes_deviation",
                                "leads_to",
                                "leads_to_consequence",
                            ],
                        )
                    elif association_query:
                        valid_chain = predicates in (
                            [
                                "leads_to_consequence",
                                "causes_deviation",
                            ],
                            [
                                "leads_to_consequence",
                                "leads_to",
                                "causes_deviation",
                            ],
                        )
                    elif direction == "object":
                        valid_chain = predicates == [
                            "leads_to",
                            "leads_to_consequence",
                        ]
                    else:
                        valid_chain = predicates == [
                            "leads_to_consequence",
                            "leads_to",
                        ]
                    if valid_chain:
                        candidates.append(next_path)

                queue.append(next_path)

        return candidates

    @staticmethod
    def _format_path(path: _Path) -> str:
        pieces = [path.display_nodes[0]]
        for edge, next_node in zip(path.edges, path.display_nodes[1:]):
            provenance = edge.origin.value
            scenario_id = getattr(edge, "scenario_id", "")
            if scenario_id:
                provenance += f", scenario_id={scenario_id}"
            if edge.source_id:
                provenance += f", source_id={edge.source_id}"
            if getattr(edge, "source_location", None):
                provenance += f", location={edge.source_location}"
            pieces.append(
                f"-[{edge.predicate} | {provenance}]-> {next_node}"
            )
        return " ".join(pieces)

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
