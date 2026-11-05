
"""Sparse TF-IDF retrieval over structured HAZOP scenario records."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable

from app.application.strategies.retrieval.models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalScope,
    RetrievedEvidence,
)
from app.domain.entities.hazop_record import (
    CausalRelation,
    HazopRecord,
    RecordType,
    RelationshipOrigin,
)
from app.domain.ontology.schema import infer_entity_types
from app.domain.repositories.graph_repository import GraphRepository

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "in", "is", "it", "of", "on", "or", "the", "to",
        "with", "due", "possible", "potential",
    }
)

_ALLOWED_CROSS_SCENARIO_RECORD_TYPES = frozenset(
    {
        RecordType.SOURCE_VERIFIED,
        RecordType.SOURCE_RECONSTRUCTED,
        RecordType.EXPERT_VALIDATED,
    }
)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in _STOP_WORDS
    ]


def _cosine(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0

    dot = sum(
        value * right.get(term, 0.0)
        for term, value in left.items()
    )

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (left_norm * right_norm)


def _scenario_context(record: HazopRecord) -> str:
    """Return target context while deliberately omitting held-out consequences."""
    sections = (
        ("Plant/unit", record.plant_unit),
        ("Equipment", record.equipment),
        ("Design intent", record.design_intent),
        ("Parameter", record.parameter),
        ("Guideword", record.guideword),
        ("Deviation", record.deviation),
        ("Known causes", "; ".join(record.causes)),
        ("Known effects", "; ".join(record.effects)),
        ("Safeguards", "; ".join(record.safeguards)),
        ("Recommendations", "; ".join(record.recommendations)),
    )
    return " | ".join(
        f"{label}: {value}"
        for label, value in sections
        if value
    )


def _record_text(record: HazopRecord) -> str:
    parts = [
        record.plant_unit,
        record.equipment,
        record.design_intent,
        record.parameter,
        record.guideword,
        record.deviation,
        *record.causes,
        *record.consequences,
        *record.safeguards,
        *record.recommendations,
    ]
    return " ".join(parts)


def _record_origin(record: HazopRecord) -> RelationshipOrigin:
    """Derive provenance from the record rather than an arbitrary relation."""

    mapping = {
        RecordType.SOURCE_VERIFIED: RelationshipOrigin.SOURCE,
        RecordType.SOURCE_RECONSTRUCTED: RelationshipOrigin.RECONSTRUCTED,
        RecordType.EXPERT_VALIDATED: RelationshipOrigin.EXPERT,
        RecordType.SYNTHETIC: RelationshipOrigin.SYNTHETIC,
        RecordType.UNSPECIFIED: RelationshipOrigin.UNSPECIFIED,
    }

    return mapping[record.record_type]


def _same_family(target: HazopRecord, candidate: HazopRecord) -> bool:
    """Check whether two records belong to the same plant/equipment family."""

    if target.plant_unit.strip():
        return (
            candidate.plant_unit.strip().casefold()
            == target.plant_unit.strip().casefold()
        )

    return (
        candidate.equipment.strip().casefold()
        == target.equipment.strip().casefold()
    )


class TfidfRetriever:
    """Retrieve row-associated cause/consequence candidates using TF-IDF.

    Retrieved relationships are marked RETRIEVAL_INFERRED. They represent
    co-occurrence-based candidates, not directly established causal edges.

    Cross-scenario candidates are restricted to the target's plant/equipment
    family and records with source-verified, reconstructed, or expert-
    validated provenance.
    """

    def __init__(
        self,
        repository: GraphRepository,
        *,
        min_score: float = 0.05,
        max_records: int = 10_000,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        if isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be positive")

        self._repository = repository
        self._min_score = min_score
        self._max_records = max_records

        self._indexes: dict[
            str,
            tuple[
                tuple[HazopRecord, ...],
                dict[str, float],
                tuple[dict[str, float], ...],
            ],
        ] = {}

        self._records: tuple[HazopRecord, ...] = ()
        self._idf: dict[str, float] = {}
        self._document_vectors: tuple[dict[str, float], ...] = ()
        self.index_build_ms = 0.0

        self.refresh()

    def refresh(self) -> None:
        """Rebuild an independent lexical index for each dataset."""

        started = time.perf_counter()

        all_records = tuple(
            self._repository.find_records(limit=self._max_records)
        )

        grouped: dict[str, list[HazopRecord]] = {}

        for record in all_records:
            grouped.setdefault(record.dataset_id, []).append(record)

        indexes: dict[
            str,
            tuple[
                tuple[HazopRecord, ...],
                dict[str, float],
                tuple[dict[str, float], ...],
            ],
        ] = {}

        for dataset_id, dataset_records in grouped.items():
            records = tuple(dataset_records)
            documents = [
                _tokens(_record_text(record))
                for record in records
            ]

            document_frequency: Counter[str] = Counter()

            for terms in documents:
                document_frequency.update(set(terms))

            count = len(documents)

            idf = {
                term: math.log((count + 1) / (frequency + 1)) + 1.0
                for term, frequency in document_frequency.items()
            }

            vectors = tuple(
                self._vector(terms, idf)
                for terms in documents
            )

            indexes[dataset_id] = (records, idf, vectors)

        self._indexes = indexes
        self._records = all_records

        default_index = indexes.get("default", ((), {}, ()))

        self._records, self._idf, self._document_vectors = (
            default_index[0],
            default_index[1],
            default_index[2],
        )

        self.index_build_ms = (
            time.perf_counter() - started
        ) * 1000.0

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        target = self._repository.get_record(
            request.scenario_id,
            dataset_id=request.dataset_id,
        )

        if target is None:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=False,
            )

        records, idf, document_vectors = self._indexes.get(
            request.dataset_id,
            ((), {}, ()),
        )

        if not records:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=True,
                scenario_context=_scenario_context(target),
            )

        query_terms = _tokens(request.known_endpoint)
        query_vector = self._vector(query_terms, idf)

        if not query_vector:
            return RetrievalResult(
                scenario_id=request.scenario_id,
                dataset_id=request.dataset_id,
                scope=RetrievalScope.SCENARIO_LOCAL,
                evidence=(),
                scenario_exists=True,
                graph_context=self._local_graph(target),
                scenario_context=_scenario_context(target),
            )

        scored_records: list[tuple[float, HazopRecord]] = []

        for record, document_vector in zip(
            records,
            document_vectors,
        ):
            if record.dataset_id != request.dataset_id:
                continue

            is_local = (
                record.scenario_id.casefold()
                == request.scenario_id.casefold()
            )

            # Cross-scenario retrieval must remain within the target family.
            if not is_local and not _same_family(target, record):
                continue

            # Synthetic and unspecified records are not cross-scenario evidence.
            if (
                not is_local
                and record.record_type
                not in _ALLOWED_CROSS_SCENARIO_RECORD_TYPES
            ):
                continue

            record_score = _cosine(query_vector, document_vector)

            if record_score > 0.0:
                scored_records.append((record_score, record))

        scored_records.sort(
            key=lambda item: (
                -item[0],
                item[1].scenario_id.casefold(),
            )
        )

        evidence: list[RetrievedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()

        for record_score, record in scored_records:
            predicate = request.predicate.strip().casefold()
            missing_endpoint = request.missing_endpoint

            # Select anchors and candidate endpoints from the semantics of the
            # requested relation. A deviation is the subject of
            # `leads_to_consequence`; it is not a cause.
            if predicate == "leads_to_consequence":
                if missing_endpoint == "object":
                    anchors = (record.deviation,) if record.deviation else ()
                    missing_values = record.consequences
                else:
                    anchors = record.consequences
                    missing_values = (
                        (record.deviation,) if record.deviation else ()
                    )

            elif predicate == "causes_deviation":
                if missing_endpoint == "object":
                    anchors = record.causes
                    missing_values = (
                        (record.deviation,) if record.deviation else ()
                    )
                else:
                    anchors = (
                        (record.deviation,) if record.deviation else ()
                    )
                    missing_values = record.causes

            else:
                # Preserve the previous cause/consequence pairing for other
                # supported co-occurrence predicates.
                anchors = (
                    record.consequences
                    if missing_endpoint == "subject"
                    else record.causes
                )
                missing_values = (
                    record.causes
                    if missing_endpoint == "subject"
                    else record.consequences
                )

            if not anchors or not missing_values:
                continue

            for anchor in anchors:
                anchor_vector = self._vector(
                    _tokens(anchor),
                    idf,
                )

                # A structured relation endpoint must match the requested
                # entity exactly. TF-IDF similarity is useful for ranking
                # records, but it must not silently turn a different entity
                # into the requested subject/object.
                if (
                    anchor.strip().casefold()
                    != request.known_endpoint.strip().casefold()
                ):
                    continue

                anchor_score = 1.0

                score = 0.75 * anchor_score + 0.25 * record_score

                if score < self._min_score:
                    continue

                for missing_value in missing_values:
                    if request.missing_endpoint == "object":
                        subject = request.known_endpoint
                        object_ = missing_value
                    else:
                        subject = missing_value
                        object_ = request.known_endpoint

                    key = (
                        record.scenario_id.casefold(),
                        subject.casefold(),
                        request.predicate.casefold(),
                        object_.casefold(),
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    row_origin = _record_origin(record)

                    provenance = (
                        record.source.provenance_note
                        if record.source is not None
                        else record.remarks
                    )

                    subject_type, object_type = infer_entity_types(
                        request.predicate,
                        subject,
                        object_,
                    )

                    relation = CausalRelation(
                        subject=subject,
                        predicate=request.predicate,
                        object=object_,
                        origin=RelationshipOrigin.RETRIEVAL_INFERRED,
                        evidence=(
                            f"TF-IDF row candidate from scenario "
                            f"{record.scenario_id}; the source row lists "
                            f"retrieved anchor {anchor!r} and candidate "
                            f"endpoint {missing_value!r} in fields associated "
                            f"with the requested relation. The query endpoint was "
                            f"{request.known_endpoint!r}. Underlying record "
                            f"type: {record.record_type.value}; relationship "
                            f"origin classification: {row_origin.value}. "
                            f"This is co-occurrence-based retrieval, not a "
                            f"directly documented causal edge. "
                            f"Provenance: {provenance}"
                        ),
                        source_id=(
                            f"{record.dataset_id}:"
                            f"{record.source.document_id}:"
                            f"{record.scenario_id}"
                            if record.source is not None
                            else (
                                f"{record.dataset_id}:"
                                f"{record.scenario_id}:"
                                f"row-{record.row_number or 0}"
                            )
                        ),
                        dataset_id=record.dataset_id,
                        subject_type=subject_type,
                        object_type=object_type,
                    )

                    evidence.append(
                        RetrievedEvidence(
                            scenario_id=record.scenario_id,
                            dataset_id=record.dataset_id,
                            relation=relation,
                            scope=(
                                RetrievalScope.SCENARIO_LOCAL
                                if record.scenario_id.casefold()
                                == request.scenario_id.casefold()
                                else RetrievalScope.CROSS_SCENARIO
                            ),
                            relevance_score=score,
                        )
                    )

            if len(evidence) >= request.limit:
                break

        evidence.sort(
            key=lambda item: (
                0
                if item.scope == RetrievalScope.SCENARIO_LOCAL
                else 1,
                -(item.relevance_score or 0.0),
                item.scenario_id.casefold(),
                item.relation.object.casefold(),
            )
        )

        evidence = evidence[: request.limit]

        scopes = {item.scope for item in evidence}

        scope = (
            RetrievalScope.HYBRID
            if len(scopes) > 1
            else next(
                iter(scopes),
                RetrievalScope.SCENARIO_LOCAL,
            )
        )

        return RetrievalResult(
            scenario_id=request.scenario_id,
            dataset_id=request.dataset_id,
            scope=scope,
            evidence=tuple(evidence),
            scenario_exists=True,
            graph_context=self._local_graph(target),
            scenario_context=_scenario_context(target),
        )

    @staticmethod
    def _vector(
        terms: Iterable[str],
        idf: dict[str, float],
    ) -> dict[str, float]:
        frequencies = Counter(terms)

        return {
            term: (1.0 + math.log(frequency)) * idf[term]
            for term, frequency in frequencies.items()
            if term in idf
        }

    @staticmethod
    def _local_graph(
        record: HazopRecord,
    ) -> tuple[RetrievedEvidence, ...]:
        return tuple(
            RetrievedEvidence(
                scenario_id=record.scenario_id,
                dataset_id=record.dataset_id,
                relation=relation,
                scope=RetrievalScope.SCENARIO_LOCAL,
            )
            for relation in record.causal_relations
            if relation.origin not in {
                RelationshipOrigin.MODEL_INFERRED,
                RelationshipOrigin.RETRIEVAL_INFERRED,
                RelationshipOrigin.GRAPH_INFERRED,
            }
        )
