
"""E3: Audited independent typed-edge evidence consensus.

Research question:
Is missing-link recovery constrained by weak support for the
cause -> deviation edge, the deviation -> consequence edge, or both?

This version records:
- Distinct training scenarios supporting every typed edge.
- Actual retrieved edge instances and their provenance.
- Candidate paths returned by GraphReasoner._find_paths().
- Paths selected by GraphReasoner._build_path_relation().
- Predictions associated with selected paths.
- Per-gold-consequence path and prediction diagnostics.

It preserves the four original E3 conditions and writes to:
    results/e3/audited/edge_support_consensus_audited.json

It does not modify E1, E2, or previous E3 outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import scripts.run_benchmark as benchmark

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.retrieval.graph_neighborhood import (
    GraphNeighborhoodRetriever,
)
from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.recover_missing_link import (
    MissingEndpoint,
    RecoverMissingLink,
    RecoveryRequest,
)


DATASET = "t301-reconstructed"
SEED = 42
TEST_FRACTION = 0.2
TOP_K = 10
MAX_HOPS = 4

CONDITIONS = {
    "edge_support_1_1": (1, 1),
    "cause_consensus_2_1": (2, 1),
    "consequence_consensus_1_2": (1, 2),
    "both_edges_consensus_2_2": (2, 2),
}

CAUSES_DEVIATION = "causes_deviation"
LEADS_TO_CONSEQUENCE = "leads_to_consequence"


def find_repository_root() -> Path:
    """Find the repository root without assuming a fixed parent depth."""

    script_path = Path(__file__).resolve()

    for parent in script_path.parents:
        if (
            (parent / "server").is_dir()
            and (parent / "results").is_dir()
        ):
            return parent

    raise RuntimeError(
        "Could not identify repository root. Expected a directory "
        "containing both 'server/' and 'results/'."
    )


ROOT = find_repository_root()


def normalize(value: Any) -> str:
    """Normalize entity labels and predicates; treat missing values as empty."""

    if value is None:
        return ""
    value = getattr(value, "value", value)
    return str(value).strip().casefold()


def relation_of(item):
    """Return the CausalRelation carried by a RetrievedEvidence wrapper."""
    return getattr(item, "relation", item)


def relation_key(item) -> tuple[str, str, str]:
    relation = relation_of(item)
    return (
        normalize(relation.subject),
        normalize(relation.predicate),
        normalize(relation.object),
    )


def serialize_retrieved_evidence(item) -> dict:
    """Serialize a RetrievedEvidence wrapper without losing its provenance."""
    relation = relation_of(item)
    serialized = serialize_edge(relation)
    serialized.update({
        "scenario_id": getattr(item, "scenario_id", serialized["scenario_id"]),
        "dataset_id": getattr(item, "dataset_id", serialized["dataset_id"]),
        "scope": getattr(
            getattr(item, "scope", None),
            "value",
            str(getattr(item, "scope", "")),
        ),
        "process_context": getattr(item, "process_context", ""),
        "relevance_score": getattr(item, "relevance_score", None),
    })
    serialized["normalized_key"] = list(relation_key(relation))
    return serialized


def same_process_family(record, target) -> bool:
    """Match the target plant unit, falling back to equipment."""

    target_plant = normalize(target.plant_unit)
    record_plant = normalize(record.plant_unit)

    if target_plant:
        return record_plant == target_plant

    target_equipment = normalize(target.equipment)
    record_equipment = normalize(record.equipment)

    return bool(target_equipment) and (
        record_equipment == target_equipment
    )


def display_prediction(value: Any) -> str:
    """Extract a comparable prediction value."""

    if value is None:
        return ""

    if isinstance(value, dict):
        for field in (
            "value",
            "candidate",
            "entity",
            "endpoint",
            "object",
            "label",
            "text",
            "name",
        ):
            if field in value and value[field] is not None:
                return display_prediction(value[field])

        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    if isinstance(value, (str, int, float)):
        return str(value).strip()

    return str(value).strip()


def prediction_values(predictions) -> list[str]:
    if predictions is None:
        return []

    if isinstance(predictions, (str, dict)):
        predictions = [predictions]

    return list(dict.fromkeys(
        display_prediction(value)
        for value in predictions
        if display_prediction(value)
    ))


def serialize_edge(edge) -> dict:
    """Serialize one EvidenceItemProtocol edge."""

    return {
        "subject": edge.subject,
        "predicate": edge.predicate,
        "object": edge.object,
        "normalized_key": list(relation_key(edge)),
        "scenario_id": getattr(edge, "scenario_id", None),
        "dataset_id": getattr(edge, "dataset_id", None),
        "source_id": getattr(edge, "source_id", None),
        "source_location": getattr(edge, "source_location", None),
        "origin": getattr(
            getattr(edge, "origin", None),
            "value",
            str(getattr(edge, "origin", "")),
        ),
        "evidence": getattr(edge, "evidence", None),
    }


def serialize_path(path) -> dict:
    """Serialize a GraphReasoner internal path."""

    return {
        "nodes": list(path.display_nodes),
        "normalized_nodes": [
            [node_type.value, value]
            for node_type, value in path.nodes
        ],
        "endpoint": path.display_endpoint,
        "normalized_endpoint": normalize(path.display_endpoint),
        "hop_count": len(path.edges),
        "edges": [
            serialize_edge(edge)
            for edge in path.edges
        ],
    }


class AuditedGraphReasoner(GraphReasoner):
    """Capture actual path discovery and path selection.

    _find_paths() records validated candidate paths.

    _build_path_relation() records paths selected by the reasoner
    for relation construction.

    These are distinct stages: a discovered path is not automatically
    a selected path or an emitted prediction.
    """

    def __init__(self, max_hops: int = 4):
        super().__init__(max_hops=max_hops)
        self.reset_audit()

    def reset_audit(self):
        self.last_candidate_paths = []
        self.last_selected_paths = []

    def _find_paths(self, context, direction):
        paths = super()._find_paths(context, direction)

        self.last_candidate_paths = [
            serialize_path(path)
            for path in paths
        ]

        return paths

    def _build_path_relation(
        self,
        path,
        *,
        context,
        dataset_id,
    ):
        relation = super()._build_path_relation(
            path,
            context=context,
            dataset_id=dataset_id,
        )

        relation_data = None

        if relation is not None:
            relation_data = {
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "origin": getattr(
                    getattr(relation, "origin", None),
                    "value",
                    str(getattr(relation, "origin", "")),
                ),
                "source_id": relation.source_id,
                "evidence": relation.evidence,
            }

        self.last_selected_paths.append({
            "path": serialize_path(path),
            "relation_constructed": relation is not None,
            "constructed_relation": relation_data,
        })

        return relation


class GlobalEdgeSupport:
    """Calculate independent edge support from eligible train records."""

    def __init__(self, train_records, test_records):
        self.train_records = tuple(train_records)

        self.targets = {
            (
                record.dataset_id.casefold(),
                record.scenario_id.casefold(),
            ): record
            for record in test_records
        }

    def target_record(self, request):
        key = (
            request.dataset_id.casefold(),
            request.scenario_id.casefold(),
        )

        target = self.targets.get(key)

        if target is None:
            raise ValueError(
                f"Target scenario not found: {request.scenario_id}"
            )

        return target

    def build(self, request):
        """Return edge -> distinct supporting scenario IDs."""

        target = self.target_record(request)
        known_cause = normalize(request.known_endpoint)

        eligible_records = [
            record
            for record in self.train_records
            if (
                record.dataset_id.casefold()
                == request.dataset_id.casefold()
                and record.scenario_id.casefold()
                != request.scenario_id.casefold()
                and same_process_family(record, target)
            )
        ]

        edge_support = defaultdict(set)

        for record in eligible_records:
            scenario = record.scenario_id.casefold()

            for relation in record.causal_relations:
                subject, predicate, obj = relation_key(relation)

                if (
                    predicate == CAUSES_DEVIATION
                    and subject == known_cause
                ):
                    edge_support[
                        (subject, predicate, obj)
                    ].add(scenario)

                elif predicate == LEADS_TO_CONSEQUENCE:
                    edge_support[
                        (subject, predicate, obj)
                    ].add(scenario)

        return edge_support, eligible_records


def support_record(edge, edge_support):
    """Return global support information for a relation or normalized key."""

    if isinstance(edge, tuple) and len(edge) == 3:
        key = tuple(normalize(value) for value in edge)
    else:
        key = relation_key(edge)

    scenarios = edge_support.get(key, set())

    return {
        "edge": {
            "subject": key[0],
            "predicate": key[1],
            "object": key[2],
        },
        "support_count": len(scenarios),
        "supporting_scenario_ids": sorted(scenarios),
    }


def enrich_path(path, edge_support, retrieved_instances,
                cause_threshold, consequence_threshold):
    """Attach support and retrieval provenance to every path edge."""

    enriched_edges = []

    for edge in path["edges"]:
        key = tuple(edge["normalized_key"])
        support = support_record(key, edge_support)

        edge_scenario = normalize(edge.get("scenario_id"))
        edge_source = normalize(edge.get("source_id"))

        matching_instances = [
            item
            for item in retrieved_instances
            if (
                tuple(item["normalized_key"]) == key
                and (
                    (
                        edge_scenario
                        and normalize(item.get("scenario_id")) == edge_scenario
                    )
                    or (
                        not edge_scenario
                        and edge_source
                        and normalize(item.get("source_id")) == edge_source
                    )
                    or (
                        not edge_scenario
                        and not edge_source
                    )
                )
            )
        ]

        predicate = key[1]

        if predicate == CAUSES_DEVIATION:
            threshold = cause_threshold
        elif predicate == LEADS_TO_CONSEQUENCE:
            threshold = consequence_threshold
        else:
            threshold = None

        enriched_edges.append({
            **edge,
            "global_support_count": support["support_count"],
            "global_supporting_scenario_ids": (
                support["supporting_scenario_ids"]
            ),
            "required_support": threshold,
            "meets_support_threshold": (
                threshold is not None
                and support["support_count"] >= threshold
            ),
            "retrieved_in_graph_context": bool(matching_instances),
            "retrieved_instances": matching_instances,
        })

    return {
        **path,
        "edges": enriched_edges,
        "all_edges_meet_support_threshold": all(
            edge["meets_support_threshold"]
            for edge in enriched_edges
        ),
        "all_edges_retrieved": all(
            edge["retrieved_in_graph_context"]
            for edge in enriched_edges
        ),
    }


class GlobalEdgeConsensusRetriever:
    """Filter graph context using independent global edge support."""

    def __init__(
        self,
        base_retriever,
        support_index: GlobalEdgeSupport,
        cause_threshold: int,
        consequence_threshold: int,
    ):
        self.base_retriever = base_retriever
        self.support_index = support_index
        self.cause_threshold = cause_threshold
        self.consequence_threshold = consequence_threshold
        self.last_audit = {}

    def retrieve(self, request):
        result = self.base_retriever.retrieve(request)

        edge_support, eligible_records = (
            self.support_index.build(request)
        )

        qualifying_cause_edges = {
            edge
            for edge, scenarios in edge_support.items()
            if (
                edge[1] == CAUSES_DEVIATION
                and len(scenarios) >= self.cause_threshold
            )
        }

        qualifying_consequence_edges = {
            edge
            for edge, scenarios in edge_support.items()
            if (
                edge[1] == LEADS_TO_CONSEQUENCE
                and len(scenarios) >= self.consequence_threshold
            )
        }

        qualifying_paths = set()

        for cause_edge in qualifying_cause_edges:
            cause, _, deviation = cause_edge

            for consequence_edge in qualifying_consequence_edges:
                deviation_subject, _, consequence = consequence_edge

                if deviation == deviation_subject:
                    qualifying_paths.add(
                        (cause, deviation, consequence)
                    )

        # Apply support thresholds independently to the two typed edge
        # families. Do not require an edge to already have a matching partner:
        # GraphReasoner determines whether retained edges compose into a path.
        qualifying_edges = (
            qualifying_cause_edges | qualifying_consequence_edges
        )

        raw_context = tuple(result.graph_context)

        # (1,1) is an exact pass-through baseline. It must reproduce ordinary
        # GraphReasoner behaviour, including graph context outside these two
        # audited edge predicates.
        if self.cause_threshold == 1 and self.consequence_threshold == 1:
            filtered_context = raw_context
        else:
            filtered_context = tuple(
                item
                for item in raw_context
                if (
                    relation_key(item) in qualifying_edges
                    and relation_key(item)[1] in {
                        CAUSES_DEVIATION,
                        LEADS_TO_CONSEQUENCE,
                    }
                )
            )

        # Retrieval candidates are not graph edges. E3 thresholds apply only
        # to graph_context, so the evidence/candidate channel stays unchanged.
        retrieved_instances = [
            serialize_retrieved_evidence(item)
            for item in raw_context
        ]

        filtered_instances = [
            serialize_retrieved_evidence(item)
            for item in filtered_context
        ]

        support_records = []

        for edge, scenarios in sorted(edge_support.items()):
            support_records.append({
                "edge": {
                    "subject": edge[0],
                    "predicate": edge[1],
                    "object": edge[2],
                },
                "support_count": len(scenarios),
                "supporting_scenario_ids": sorted(scenarios),
                "required_support": (
                    self.cause_threshold
                    if edge[1] == CAUSES_DEVIATION
                    else self.consequence_threshold
                ),
                "qualifies": (
                    len(scenarios)
                    >= (
                        self.cause_threshold
                        if edge[1] == CAUSES_DEVIATION
                        else self.consequence_threshold
                    )
                ),
            })

        path_records = []

        for cause, deviation, consequence in sorted(qualifying_paths):
            cause_edge = (cause, CAUSES_DEVIATION, deviation)
            consequence_edge = (
                deviation,
                LEADS_TO_CONSEQUENCE,
                consequence,
            )

            path_records.append({
                "cause": cause,
                "deviation": deviation,
                "consequence": consequence,
                "cause_edge": support_record(
                    cause_edge,
                    edge_support,
                ),
                "consequence_edge": support_record(
                    consequence_edge,
                    edge_support,
                ),
            })

        cause_histogram = defaultdict(int)
        consequence_histogram = defaultdict(int)

        for edge, scenarios in edge_support.items():
            if edge[1] == CAUSES_DEVIATION:
                cause_histogram[len(scenarios)] += 1
            elif edge[1] == LEADS_TO_CONSEQUENCE:
                consequence_histogram[len(scenarios)] += 1

        self.last_audit = {
            "cause_support_threshold": self.cause_threshold,
            "consequence_support_threshold": (
                self.consequence_threshold
            ),
            "eligible_training_scenarios": len(eligible_records),
            "eligible_training_scenario_ids": sorted({
                record.scenario_id
                for record in eligible_records
            }),
            "global_cause_edge_count": sum(
                edge[1] == CAUSES_DEVIATION
                for edge in edge_support
            ),
            "global_consequence_edge_count": sum(
                edge[1] == LEADS_TO_CONSEQUENCE
                for edge in edge_support
            ),
            "qualifying_cause_edge_count": len(
                qualifying_cause_edges
            ),
            "qualifying_consequence_edge_count": len(
                qualifying_consequence_edges
            ),
            "qualifying_path_count": len(qualifying_paths),
            "cause_support_histogram": dict(
                sorted(cause_histogram.items())
            ),
            "consequence_support_histogram": dict(
                sorted(consequence_histogram.items())
            ),
            "raw_retrieved_edge_count": len(raw_context),
            "filtered_retrieved_edge_count": len(filtered_context),
            "retrieved_edge_instance_count": len(retrieved_instances),
            "filtered_edge_instance_count": len(filtered_instances),
            "baseline_passthrough": (
                self.cause_threshold == 1
                and self.consequence_threshold == 1
            ),
            "retrieved_edge_instances": retrieved_instances,
            "filtered_edge_instances": filtered_instances,
            "global_edge_support": support_records,
            "qualifying_paths": path_records,
        }

        return replace(
            result,
            graph_context=filtered_context,
        )


def enrich_reasoning_trace(
    reasoner,
    retriever,
    predictions,
    gold_values,
):
    """Join actual reasoner paths with edge support and predictions."""

    audit = retriever.last_audit

    edge_support = {}

    for record in audit.get("global_edge_support", []):
        edge = record["edge"]
        key = (
            normalize(edge["subject"]),
            normalize(edge["predicate"]),
            normalize(edge["object"]),
        )
        edge_support[key] = {
            "support_count": record["support_count"],
            "supporting_scenario_ids": (
                record["supporting_scenario_ids"]
            ),
            "required_support": record["required_support"],
            "qualifies": record["qualifies"],
        }

    retrieved_instances = audit.get(
        "retrieved_edge_instances",
        [],
    )

    cause_threshold = audit["cause_support_threshold"]
    consequence_threshold = audit["consequence_support_threshold"]

    candidate_paths = [
        enrich_path(
            path,
            edge_support,
            retrieved_instances,
            cause_threshold,
            consequence_threshold,
        )
        for path in reasoner.last_candidate_paths
    ]

    selected_paths = []

    prediction_map = {
        normalize(value): value
        for value in prediction_values(predictions)
    }

    for selected in reasoner.last_selected_paths:
        path = enrich_path(
            selected["path"],
            edge_support,
            retrieved_instances,
            cause_threshold,
            consequence_threshold,
        )

        relation = selected["constructed_relation"]

        endpoint = (
            relation.get("object")
            if relation
            else path["endpoint"]
        )

        normalized_endpoint = normalize(endpoint)

        matched_prediction = prediction_map.get(
            normalized_endpoint
        )

        selected_paths.append({
            **selected,
            "path": path,
            "endpoint": endpoint,
            "prediction_emitted": matched_prediction is not None,
            "matched_prediction": matched_prediction,
        })

    gold_diagnostics = []

    for gold in gold_values:
        normalized_gold = normalize(gold)

        candidate_matches = [
            path
            for path in candidate_paths
            if path["normalized_endpoint"] == normalized_gold
        ]

        selected_matches = [
            path
            for path in selected_paths
            if normalize(path["endpoint"]) == normalized_gold
        ]

        gold_diagnostics.append({
            "gold_consequence": gold,
            "predicted": normalized_gold in prediction_map,
            "candidate_path_count": len(candidate_matches),
            "selected_path_count": len(selected_matches),
            "candidate_paths": candidate_matches,
            "selected_paths": selected_matches,
            "failure_stage": (
                "prediction_emitted"
                if normalized_gold in prediction_map
                else (
                    "path_selected_but_not_emitted"
                    if selected_matches
                    else (
                        "candidate_path_found_but_not_selected"
                        if candidate_matches
                        else "no_candidate_path_found"
                    )
                )
            ),
        })

    return {
        "candidate_path_count": len(candidate_paths),
        "selected_path_count": len(selected_paths),
        "candidate_paths": candidate_paths,
        "selected_paths": selected_paths,
        "gold_diagnostics": gold_diagnostics,
    }


def run_condition(
    *,
    condition_name,
    cause_threshold,
    consequence_threshold,
    repository,
    support_index,
    cases,
    builder,
):
    local_retriever = ScenarioLocalRetriever(repository)

    graph_retriever = GraphNeighborhoodRetriever(
        local_retriever,
        repository,
    )

    consensus_retriever = GlobalEdgeConsensusRetriever(
        base_retriever=graph_retriever,
        support_index=support_index,
        cause_threshold=cause_threshold,
        consequence_threshold=consequence_threshold,
    )

    use_case = RecoverMissingLink(consensus_retriever)

    # A fresh audited reasoner is used for each condition.
    reasoner = AuditedGraphReasoner(max_hops=MAX_HOPS)

    evaluated_cases = []

    print(
        f"\n[E3 Audited] {condition_name} "
        f"| cause>={cause_threshold} "
        f"| consequence>={consequence_threshold}"
    )

    for index, case in enumerate(cases, start=1):
        request = RecoveryRequest(
            scenario_id=case.scenario_id,
            predicate="co_listed_with_consequence",
            known_endpoint=case.known_endpoint,
            missing_endpoint=MissingEndpoint.OBJECT,
            limit=TOP_K,
            dataset_id=case.dataset_id,
        )

        reasoner.reset_audit()

        predictions, latency, status, diagnostics = (
            benchmark._run_recovery(
                use_case,
                builder,
                request,
                reasoner=reasoner,
            )
        )

        # Keep benchmark prediction dictionaries intact: score_predictions()
        # expects prediction["value"], origin, and evidence. Convert to strings
        # only inside the path-to-prediction diagnostic join.
        predictions = list(predictions or [])

        edge_audit = dict(consensus_retriever.last_audit)

        reasoning_audit = enrich_reasoning_trace(
            reasoner=reasoner,
            retriever=consensus_retriever,
            predictions=predictions,
            gold_values=case.gold_values,
        )

        evaluated_cases.append({
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "known_endpoint": case.known_endpoint,
            "gold_values": list(case.gold_values),
            "predictions": predictions,
            "latency_ms": latency,
            "status": status,
            "diagnostics": diagnostics,
            "edge_support_audit": edge_audit,
            "reasoning_trace": reasoning_audit,
        })

        print(
            f"[E3 Audited] {index}/{len(cases)} "
            f"{case.case_id} | "
            f"global_paths={edge_audit['qualifying_path_count']} | "
            f"retrieved={edge_audit['raw_retrieved_edge_count']} | "
            f"filtered={edge_audit['filtered_retrieved_edge_count']} | "
            f"candidate_paths={reasoning_audit['candidate_path_count']} | "
            f"selected_paths={reasoning_audit['selected_path_count']} | "
            f"predictions={len(predictions)}"
        )

    return {
        "condition": condition_name,
        "cause_support_threshold": cause_threshold,
        "consequence_support_threshold": consequence_threshold,
        "summary": benchmark.score_predictions(evaluated_cases),
        "cases": evaluated_cases,
    }


def build_comparison(conditions):
    """Create case-level gold transitions against the (1,1) condition."""

    baseline = conditions["edge_support_1_1"]["cases"]

    comparisons = {}

    for condition_name, condition in conditions.items():
        if condition_name == "edge_support_1_1":
            continue

        comparison_by_id = {
            case["case_id"]: case
            for case in condition["cases"]
        }

        rows = []

        for base_case in baseline:
            other = comparison_by_id[base_case["case_id"]]

            gold = {
                normalize(value): value
                for value in base_case["gold_values"]
            }

            base_predictions = {
                normalize(item.get("value"))
                for item in base_case["predictions"]
                if isinstance(item, dict) and normalize(item.get("value"))
            }

            other_predictions = {
                normalize(item.get("value"))
                for item in other["predictions"]
                if isinstance(item, dict) and normalize(item.get("value"))
            }

            lost = (set(gold) & base_predictions) - other_predictions
            retained = set(gold) & base_predictions & other_predictions
            newly_recovered = (
                set(gold) & other_predictions
            ) - base_predictions

            rows.append({
                "case_id": base_case["case_id"],
                "scenario_id": base_case["scenario_id"],
                "gold_lost": [
                    gold[key] for key in sorted(lost)
                ],
                "gold_retained": [
                    gold[key] for key in sorted(retained)
                ],
                "gold_newly_recovered": [
                    gold[key] for key in sorted(newly_recovered)
                ],
                "baseline_predictions": base_case["predictions"],
                "comparison_predictions": other["predictions"],
                "baseline_gold_diagnostics": (
                    base_case["reasoning_trace"]["gold_diagnostics"]
                ),
                "comparison_gold_diagnostics": (
                    other["reasoning_trace"]["gold_diagnostics"]
                ),
            })

        comparisons[condition_name] = rows

    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Run audited E3 edge-support consensus."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / "e3"
            / "audited"
            / "edge_support_consensus_audited_v2.json"
        ),
        help="Output JSON path.",
    )

    args = parser.parse_args()

    output_path = args.output.expanduser().resolve()

    print(f"[E3 Audited] Repository root: {ROOT}")
    print("[E3 Audited] Loading reconstructed dataset...")

    records = benchmark.load_dataset(DATASET, None)

    train, test = benchmark.stable_split(
        records,
        test_fraction=TEST_FRACTION,
        seed=SEED,
    )

    cases = benchmark.make_cases(test)

    print(f"[E3 Audited] Records: {len(records)}")
    print(f"[E3 Audited] Training scenarios: {len(train)}")
    print(f"[E3 Audited] Test scenarios: {len(test)}")
    print(f"[E3 Audited] Test queries: {len(cases)}")

    repository = benchmark.build_repository(train, test)

    support_index = GlobalEdgeSupport(train, test)
    builder = BuildEvidenceContext()

    conditions = {}

    for name, (cause_threshold, consequence_threshold) in (
        CONDITIONS.items()
    ):
        conditions[name] = run_condition(
            condition_name=name,
            cause_threshold=cause_threshold,
            consequence_threshold=consequence_threshold,
            repository=repository,
            support_index=support_index,
            cases=cases,
            builder=builder,
        )

    comparisons = build_comparison(conditions)

    dataset_path = benchmark.DATASET_PATHS[DATASET]

    dataset_hash = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()

    output = {
        "metadata": {
            "experiment": "E3",
            "title": "Audited independent typed-edge evidence consensus",
            "protocol_version": "E3-2.1",
            "research_question": (
                "Is missing-link recovery constrained by weak support "
                "for the cause-to-deviation edge, the "
                "deviation-to-consequence edge, or both?"
            ),
            "dataset": DATASET,
            "dataset_sha256": dataset_hash,
            "dataset_records": len(records),
            "train_scenarios": len(train),
            "test_scenarios": len(test),
            "test_queries": len(cases),
            "seed": SEED,
            "test_fraction": TEST_FRACTION,
            "top_k": TOP_K,
            "max_graph_hops": MAX_HOPS,
            "conditions": {
                name: {
                    "cause_support_threshold": values[0],
                    "consequence_support_threshold": values[1],
                }
                for name, values in CONDITIONS.items()
            },
            "test_scenario_ids": sorted(
                record.scenario_id for record in test
            ),
            "case_ids": [case.case_id for case in cases],
            "path_definition": (
                "Cause --causes_deviation--> Deviation "
                "--leads_to_consequence--> Consequence"
            ),
            "support_definition": (
                "Distinct eligible training scenario IDs supporting "
                "each typed edge independently. Cause-edge support is "
                "restricted to the query's known cause; consequence-edge "
                "support is computed over the eligible process family."
            ),
            "path_composition": (
                "The two edges may be supported by different scenarios. "
                "Thresholds are applied independently to the two audited "
                "graph-context edge types. Other graph predicates are "
                "excluded in thresholded conditions; GraphReasoner "
                "determines whether retained edges form a valid typed path. "
                "The (1,1) condition is an exact unfiltered graph-context "
                "pass-through baseline."
            ),
            "reasoning_instrumentation": {
                "candidate_paths": (
                    "Captured from GraphReasoner._find_paths(). "
                    "These are validated paths discovered in the "
                    "filtered graph context."
                ),
                "selected_paths": (
                    "Captured from GraphReasoner._build_path_relation(). "
                    "These are paths selected for relation construction."
                ),
                "prediction_matching": (
                    "A selected path is matched to emitted predictions "
                    "using its constructed relation object endpoint."
                ),
            },
            "support_scope": (
                "All eligible training records in the target plant unit; "
                "equipment fallback when plant unit is unavailable."
            ),
            "llm_enabled": False,
            "label_warning": (
                "Reconstructed row-level associations are silver labels, "
                "not expert-validated causal ground truth."
            ),
            "independence_warning": (
                "Distinct scenario IDs may share source templates "
                "and assumptions; they do not establish statistical "
                "independence or physical confirmation."
            ),
        },
        "conditions": conditions,
        "case_comparisons_against_baseline": comparisons,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== E3 AUDITED RESULTS ==========")

    for name, condition in conditions.items():
        metrics = condition["summary"]

        print(
            f"{name:32s} | "
            f"F1={metrics['micro_f1']:.4f} | "
            f"P={metrics['micro_precision']:.4f} | "
            f"R={metrics['micro_recall']:.4f} | "
            f"Coverage={metrics['coverage']:.4f} | "
            f"Abstention={metrics['abstention_rate']:.4f}"
        )

    print("\n========== CASE-LEVEL GOLD LOSSES ==========")

    for name, rows in comparisons.items():
        lost = sum(
            len(row["gold_lost"])
            for row in rows
        )

        retained = sum(
            len(row["gold_retained"])
            for row in rows
        )

        print(
            f"{name:32s} | "
            f"gold_lost={lost} | "
            f"gold_retained={retained}"
        )

    print(f"\n[E3 Audited] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
