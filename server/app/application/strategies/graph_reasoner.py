"""Graph reasoning over typed, provenance-aware HAZOP relationships."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from app.domain.entities.hazop_record import (
    CausalRelation,
    RelationshipOrigin,
)
from app.domain.ontology.schema import (
    EntityType,
    RELATION_SCHEMA,
    validate_relation_types,
)
from app.domain.reasoning.models import (
    AbstentionReason,
    EvidenceContextProtocol,
    EvidenceItemProtocol,
    ReasoningResult,
    ReasoningStatus,
)


NodeKey = tuple[EntityType, str]


@dataclass(frozen=True, slots=True)
class _Path:
    """A directed path through graph evidence."""

    nodes: tuple[NodeKey, ...]
    display_nodes: tuple[str, ...]
    edges: tuple[EvidenceItemProtocol, ...]

    @property
    def endpoint(self) -> NodeKey:
        """Return the normalized terminal node."""
        return self.nodes[-1]

    @property
    def display_endpoint(self) -> str:
        """Return the original-case terminal node."""
        return self.display_nodes[-1]


class GraphReasoner:
    """Recover candidates using direct edges or typed multi-hop paths.

    Direct evidence matching the requested predicate is preferred.

    When direct evidence is unavailable, bounded graph traversal can compose
    explicitly typed causal relationships.

    Path-derived results are marked GRAPH_INFERRED and retain the complete
    provenance path. They represent indirect reachability or row-association
    hypotheses, not independently verified direct causal relationships.
    """

    # Experimental record relationships are eligible regardless of record
    # type. Their original provenance is preserved in the resulting evidence.
    #
    # Previously inferred relationships are excluded from graph expansion
    # to prevent recursive inference and circular evidence reinforcement.
    _DIRECT_ORIGINS = frozenset(
        {
            RelationshipOrigin.SOURCE,
            RelationshipOrigin.RECONSTRUCTED,
            RelationshipOrigin.SYNTHETIC,
            RelationshipOrigin.EXPERT,
            RelationshipOrigin.UNSPECIFIED,
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

    _ASSOCIATION_PREDICATE = "co_listed_with_consequence"

    def __init__(self, max_hops: int = 4) -> None:
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise TypeError("max_hops must be an integer")

        if max_hops < 2:
            raise ValueError("max_hops must be at least 2")

        self._max_hops = max_hops

    def reason(
        self,
        context: EvidenceContextProtocol,
    ) -> ReasoningResult:
        """Propose a unique direct candidate or graph-path endpoint."""

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
                scenario_id=context.scenario_id,
                candidates=direct,
                direction=direction,
                dataset_id=dataset_id,
            )

        paths = self._find_paths(context, direction)

        if not paths:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
            )

        by_endpoint: dict[NodeKey, list[_Path]] = defaultdict(list)

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
        selected = min(
            selected_paths,
            key=lambda path: len(path.edges),
        )

        relation = self._build_path_relation(
            selected,
            context=context,
            dataset_id=dataset_id,
        )

        if relation is None:
            return self._abstain(
                context.scenario_id,
                AbstentionReason.INSUFFICIENT_SUPPORT,
                dataset_id,
            )

        return ReasoningResult(
            status=ReasoningStatus.PROPOSED,
            scenario_id=context.scenario_id,
            dataset_id=dataset_id,
            proposed_relation=relation,
            supporting_evidence=tuple(
                self._format_path(path)
                for path in selected_paths
            ),
            confidence=None,
        )

    def reason_all(
        self,
        context: EvidenceContextProtocol,
    ) -> tuple[CausalRelation, ...]:
        """Return all supported endpoints for association recovery.

        This method is specifically for set-valued consequence recovery.
        It does not claim that graph-composed paths are direct causal edges.
        """

        if (
            context.predicate.strip().casefold()
            != self._ASSOCIATION_PREDICATE
        ):
            return ()

        if not context.graph_context:
            return ()

        direction = context.missing_endpoint.strip().casefold()

        if direction not in {"subject", "object"}:
            return ()

        # Association recovery requires the scenario's deviation context.
        deviation = self._extract_context_deviation(
            context.scenario_context,
        )

        if not deviation:
            return ()

        paths = self._find_paths(context, direction)

        if not paths:
            return ()

        requested_spec = RELATION_SCHEMA.get(
            self._ASSOCIATION_PREDICATE,
        )

        if requested_spec is None:
            return ()

        by_endpoint: dict[NodeKey, list[_Path]] = defaultdict(list)

        for path in paths:
            by_endpoint[path.endpoint].append(path)

        dataset_id = getattr(context, "dataset_id", "default")
        relations: list[CausalRelation] = []

        for endpoint_paths in by_endpoint.values():
            selected = min(
                endpoint_paths,
                key=lambda path: len(path.edges),
            )

            relation = self._build_path_relation(
                selected,
                context=context,
                dataset_id=dataset_id,
            )

            if relation is None:
                continue

            if (
                relation.subject_type not in requested_spec.subject_types
                or relation.object_type not in requested_spec.object_types
            ):
                continue

            relations.append(relation)

        return tuple(relations)

    def _direct_candidates(
        self,
        context: EvidenceContextProtocol,
        direction: str,
    ) -> list[EvidenceItemProtocol]:
        """Find unique, eligible direct evidence candidates."""

        predicate = context.predicate.strip().casefold()
        known = context.known_endpoint.strip().casefold()
        dataset_id = getattr(context, "dataset_id", "default")

        unique: dict[
            tuple[str, str, str, str, str | None],
            EvidenceItemProtocol,
        ] = {}

        for item in context.evidence_items:
            if item.dataset_id != dataset_id:
                continue

            if item.origin not in self._DIRECT_ORIGINS:
                continue

            if (
                item.origin == RelationshipOrigin.EXPERT
                and not item.is_expert_verified
            ):
                continue

            if not item.evidence.strip():
                continue

            if item.predicate.strip().casefold() != predicate:
                continue

            if not item.subject.strip() or not item.object.strip():
                continue

            endpoint_matches = (
                item.object.strip().casefold() == known
                if direction == "subject"
                else item.subject.strip().casefold() == known
            )

            if not endpoint_matches:
                continue

            subject_type = EntityType(
                getattr(item, "subject_type", EntityType.CONCEPT)
            )
            object_type = EntityType(
                getattr(item, "object_type", EntityType.CONCEPT)
            )

            if validate_relation_types(
                item.predicate,
                subject_type,
                object_type,
            ) is not None:
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
        *,
        scenario_id: str,
        candidates: list[EvidenceItemProtocol],
        direction: str,
        dataset_id: str,
    ) -> ReasoningResult:
        """Resolve direct candidates or abstain when ambiguous."""

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
            dataset_id=selected.dataset_id,
            is_expert_verified=selected.is_expert_verified,
            subject_type=EntityType(
                getattr(selected, "subject_type", EntityType.CONCEPT)
            ),
            object_type=EntityType(
                getattr(selected, "object_type", EntityType.CONCEPT)
            ),
            source_location=getattr(
                selected,
                "source_location",
                None,
            ),
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
                dict.fromkeys(
                    item.evidence for item in candidates
                )
            ),
            confidence=max(confidences) if confidences else None,
        )

    def _find_paths(
        self,
        context: EvidenceContextProtocol,
        direction: str,
    ) -> list[_Path]:
        """Find bounded, typed, non-cyclic paths through graph evidence."""

        known = context.known_endpoint.strip().casefold()
        dataset_id = getattr(context, "dataset_id", "default")

        adjacency: dict[
            NodeKey,
            list[tuple[NodeKey, EvidenceItemProtocol, str]],
        ] = defaultdict(list)

        start_nodes: list[NodeKey] = []

        for edge in context.graph_context:
            if edge.dataset_id != dataset_id:
                continue

            if edge.origin not in self._DIRECT_ORIGINS:
                continue

            if (
                edge.origin == RelationshipOrigin.EXPERT
                and not edge.is_expert_verified
            ):
                continue

            if not edge.evidence.strip():
                continue

            if (
                edge.predicate.strip().casefold()
                not in self._CAUSAL_PREDICATES
            ):
                continue

            if not edge.subject.strip() or not edge.object.strip():
                continue

            subject_type = EntityType(
                getattr(edge, "subject_type", EntityType.CONCEPT)
            )
            object_type = EntityType(
                getattr(edge, "object_type", EntityType.CONCEPT)
            )

            if validate_relation_types(
                edge.predicate,
                subject_type,
                object_type,
            ) is not None:
                continue

            subject: NodeKey = (
                subject_type,
                edge.subject.strip().casefold(),
            )
            object_: NodeKey = (
                object_type,
                edge.object.strip().casefold(),
            )

            if direction == "object":
                adjacency[subject].append(
                    (object_, edge, edge.object)
                )

                if subject[1] == known:
                    start_nodes.append(subject)

            else:
                adjacency[object_].append(
                    (subject, edge, edge.subject)
                )

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

        association_query = (
            context.predicate.strip().casefold()
            == self._ASSOCIATION_PREDICATE
        )

        deviation = (
            self._extract_context_deviation(
                context.scenario_context,
            )
            if association_query
            else None
        )

        if association_query and not deviation:
            return []

        while queue:
            path = queue.popleft()

            if len(path.edges) >= self._max_hops:
                continue

            current = path.nodes[-1]

            for neighbour, edge, display_value in adjacency.get(
                current,
                [],
            ):
                if neighbour in path.nodes:
                    continue

                next_path = _Path(
                    nodes=(*path.nodes, neighbour),
                    display_nodes=(*path.display_nodes, display_value),
                    edges=(*path.edges, edge),
                )

                if len(next_path.edges) >= 2:
                    predicates = [
                        item.predicate.strip().casefold()
                        for item in next_path.edges
                    ]

                    if association_query:
                        valid_chain = self._is_association_chain(
                            predicates,
                            direction,
                        )

                        if valid_chain and deviation:
                            deviation_nodes = {
                                value
                                for entity_type, value in next_path.nodes
                                if entity_type == EntityType.DEVIATION
                            }

                            valid_chain = deviation in deviation_nodes

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
    def _is_association_chain(
        predicates: list[str],
        direction: str,
    ) -> bool:
        """Validate supported forward or reverse association path patterns."""

        if direction == "object":
            return predicates in (
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

        return predicates in (
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

    @staticmethod
    def _extract_context_deviation(
        scenario_context: str,
    ) -> str | None:
        """Extract the deviation from the scenario context representation."""

        if not scenario_context.strip():
            return None

        parts = [
            part.strip()
            for part in scenario_context.split(" | ")
            if part.strip()
        ]

        # New contexts use explicit labels and may include causes, effects,
        # safeguards, and recommendations after the deviation. Read the
        # labeled deviation rather than assuming it is the final field.
        for part in parts:
            label, separator, value = part.partition(":")
            if separator and label.strip().casefold() == "deviation":
                return value.strip().casefold() or None

        # Backward compatibility with older unlabeled context strings.
        return parts[-1].casefold() if parts else None

    def _build_path_relation(
        self,
        path: _Path,
        *,
        context: EvidenceContextProtocol,
        dataset_id: str,
    ) -> CausalRelation | None:
        """Build a provenance-preserving relation from a graph path."""

        direction = context.missing_endpoint.strip().casefold()

        if direction not in {"subject", "object"}:
            return None

        known = context.known_endpoint.strip()
        endpoint = path.display_endpoint

        known_type = path.nodes[0][0]
        endpoint_type = path.nodes[-1][0]

        if direction == "subject":
            subject, object_ = endpoint, known
            subject_type, object_type = endpoint_type, known_type
        else:
            subject, object_ = known, endpoint
            subject_type, object_type = known_type, endpoint_type

        predicate = context.predicate.strip()
        requested_spec = RELATION_SCHEMA.get(predicate.casefold())

        if requested_spec is None:
            return None

        if (
            subject_type not in requested_spec.subject_types
            or object_type not in requested_spec.object_types
        ):
            return None

        if validate_relation_types(
            predicate,
            subject_type,
            object_type,
        ) is not None:
            return None

        path_evidence = self._format_path(path)

        path_source_ids = [
            edge.source_id
            for edge in path.edges
            if edge.source_id
        ]

        if predicate.casefold() == self._ASSOCIATION_PREDICATE:
            evidence = (
                "Row-association candidate inferred by typed graph "
                f"composition. {path_evidence} The endpoints are co-listed "
                "through a reconstructed relationship path; this does not "
                "establish an independently validated causal assertion."
            )
        else:
            evidence = (
                "Indirect causal reachability inferred by typed graph "
                f"composition. {path_evidence} This is not evidence that "
                "the endpoints are an immediate/direct cause-effect pair."
            )

        source_id = (
            f"graph-path:{dataset_id}:" + "->".join(path_source_ids)
            if path_source_ids
            else None
        )

        return CausalRelation(
            subject=subject,
            predicate=predicate,
            object=object_,
            origin=RelationshipOrigin.GRAPH_INFERRED,
            evidence=evidence,
            confidence=None,
            source_id=source_id,
            dataset_id=dataset_id,
            subject_type=subject_type,
            object_type=object_type,
        )

    @staticmethod
    def _format_path(path: _Path) -> str:
        """Format the complete path with edge provenance."""

        pieces = [path.display_nodes[0]]

        for edge, next_node in zip(
            path.edges,
            path.display_nodes[1:],
        ):
            provenance = edge.origin.value

            scenario_id = getattr(edge, "scenario_id", "")

            if scenario_id:
                provenance += f", scenario_id={scenario_id}"

            if edge.source_id:
                provenance += f", source_id={edge.source_id}"

            source_location = getattr(
                edge,
                "source_location",
                None,
            )

            if source_location:
                provenance += f", location={source_location}"

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
        """Create a structured abstention result."""

        return ReasoningResult(
            status=ReasoningStatus.ABSTAINED,
            scenario_id=scenario_id,
            dataset_id=dataset_id,
            abstention_reason=reason,
        )
