
"""E2: Global complete-path evidence consensus.

Research question:
How does the number of distinct training scenarios supporting a
complete typed path affect missing-link recovery?

Path:
    Cause --causes_deviation--> Deviation
          --leads_to_consequence--> Consequence

Support is calculated globally from all eligible training records,
not from the bounded graph-retrieval result.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
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


ROOT = Path(__file__).resolve().parents[2]

DATASET = "t301-reconstructed"
SEED = 42
TEST_FRACTION = 0.2
TOP_K = 10
MAX_HOPS = 4

SUPPORT_THRESHOLDS = (1, 2, 3, 4, 5, 6)

CAUSES_DEVIATION = "causes_deviation"
LEADS_TO_CONSEQUENCE = "leads_to_consequence"


def normalize(value: Any) -> str:
    """Normalize entity labels and predicates."""
    value = getattr(value, "value", value)
    return str(value).strip().casefold()


def relation_key(relation) -> tuple[str, str, str]:
    """Return a normalized typed relationship identity."""
    return (
        normalize(relation.subject),
        normalize(relation.predicate),
        normalize(relation.object),
    )


def evidence_key(item) -> tuple[str, str, str, str]:
    """Return a retrieved edge identity including its source scenario."""
    subject, predicate, obj = relation_key(item.relation)

    return (
        item.scenario_id.strip().casefold(),
        subject,
        predicate,
        obj,
    )


def same_process_family(record, target) -> bool:
    """Match the target process family for global support counting."""

    target_plant = normalize(target.plant_unit)
    record_plant = normalize(record.plant_unit)

    if target_plant:
        return record_plant == target_plant

    target_equipment = normalize(target.equipment)
    record_equipment = normalize(record.equipment)

    return bool(target_equipment) and (
        record_equipment == target_equipment
    )


class GlobalPathSupport:
    """Build complete-path support counts from all eligible train records."""

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
        """Build support index for one known cause and target process family.

        Returns:
            path_support:
                Complete path -> set of supporting scenario IDs.

            path_edges:
                Complete path -> scenario ID -> its two typed edge keys.

            eligible_scenarios:
                Number of eligible training scenarios examined.
        """

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

        path_support = defaultdict(set)
        path_edges = defaultdict(lambda: defaultdict(set))

        for record in eligible_records:
            relations = record.causal_relations

            cause_edges = []
            consequence_edges = []

            for relation in relations:
                subject, predicate, obj = relation_key(relation)

                if (
                    predicate == CAUSES_DEVIATION
                    and subject == known_cause
                ):
                    cause_edges.append(
                        (subject, predicate, obj)
                    )

                elif predicate == LEADS_TO_CONSEQUENCE:
                    consequence_edges.append(
                        (subject, predicate, obj)
                    )

            # Join only edges belonging to the same source record.
            for cause_edge in cause_edges:
                cause, _, deviation = cause_edge

                for consequence_edge in consequence_edges:
                    deviation_subject, _, consequence = (
                        consequence_edge
                    )

                    if deviation != deviation_subject:
                        continue

                    path = (
                        cause,
                        deviation,
                        consequence,
                    )

                    scenario = record.scenario_id.casefold()

                    # A scenario contributes at most one vote per path.
                    path_support[path].add(scenario)

                    path_edges[path][scenario].add(cause_edge)
                    path_edges[path][scenario].add(
                        consequence_edge
                    )

        return (
            path_support,
            path_edges,
            len(eligible_records),
        )


class GlobalPathConsensusRetriever:
    """Apply globally computed path support to retrieved graph evidence."""

    def __init__(
        self,
        base_retriever,
        support_index: GlobalPathSupport,
        min_support: int,
    ):
        if min_support < 1:
            raise ValueError("min_support must be >= 1")

        self.base_retriever = base_retriever
        self.support_index = support_index
        self.min_support = min_support
        self.last_audit = {}

    def retrieve(self, request):
        result = self.base_retriever.retrieve(request)

        (
            path_support,
            path_edges,
            eligible_scenarios,
        ) = self.support_index.build(request)

        qualifying_paths = {
            path
            for path, scenarios in path_support.items()
            if len(scenarios) >= self.min_support
        }

        support_histogram = Counter(
            len(scenarios)
            for scenarios in path_support.values()
        )

        # Construct the complete set of globally supported source edges.
        globally_supported_edges = set()

        for path in qualifying_paths:
            for scenario, edges in path_edges[path].items():
                for edge in edges:
                    globally_supported_edges.add(
                        (scenario, *edge)
                    )

        raw_context = tuple(result.graph_context)

        # Threshold 1 is an exact passthrough.
        # This preserves the original graph-reasoning baseline.
        if self.min_support == 1:
            filtered_context = raw_context
            filtered_evidence = tuple(result.evidence)

        else:
            filtered_context = tuple(
                item
                for item in raw_context
                if evidence_key(item) in globally_supported_edges
            )

            filtered_evidence = tuple(
                item
                for item in result.evidence
                if evidence_key(item) in globally_supported_edges
            )

        self.last_audit = {
            "minimum_path_support": self.min_support,
            "eligible_training_scenarios": eligible_scenarios,
            "global_complete_paths": len(path_support),
            "global_qualifying_paths": len(qualifying_paths),
            "maximum_global_path_support": max(
                (len(scenarios) for scenarios in path_support.values()),
                default=0,
            ),
            "global_path_support_histogram": {
                str(count): frequency
                for count, frequency in sorted(
                    support_histogram.items()
                )
            },
            "raw_retrieved_edges": len(raw_context),
            "retained_retrieved_edges": len(filtered_context),
            "filter_applied": self.min_support > 1,
        }

        return replace(
            result,
            evidence=filtered_evidence,
            graph_context=filtered_context,
        )


def run_condition(
    *,
    threshold: int,
    repository,
    support_index,
    cases,
    builder,
    reasoner,
):
    local_retriever = ScenarioLocalRetriever(repository)

    graph_retriever = GraphNeighborhoodRetriever(
        local_retriever,
        repository,
    )

    consensus_retriever = GlobalPathConsensusRetriever(
        base_retriever=graph_retriever,
        support_index=support_index,
        min_support=threshold,
    )

    use_case = RecoverMissingLink(consensus_retriever)

    evaluated_cases = []

    print(
        f"\n[E2] Global path-support threshold: {threshold}"
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

        predictions, latency, status, diagnostics = (
            benchmark._run_recovery(
                use_case,
                builder,
                request,
                reasoner=reasoner,
            )
        )

        audit = dict(consensus_retriever.last_audit)

        evaluated_cases.append({
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "known_endpoint": case.known_endpoint,
            "gold_values": list(case.gold_values),
            "predictions": predictions,
            "latency_ms": latency,
            "status": status,
            "diagnostics": diagnostics,
            "global_path_audit": audit,
        })

        print(
            f"[E2] {index}/{len(cases)} "
            f"{case.case_id} | "
            f"eligible={audit['eligible_training_scenarios']} | "
            f"paths={audit['global_complete_paths']} | "
            f"qualifying={audit['global_qualifying_paths']} | "
            f"edges={audit['retained_retrieved_edges']}/"
            f"{audit['raw_retrieved_edges']} | "
            f"predictions={len(predictions)}"
        )

    return {
        "threshold": threshold,
        "summary": benchmark.score_predictions(evaluated_cases),
        "cases": evaluated_cases,
    }


def main():
    print("[E2] Loading reconstructed dataset...")

    records = benchmark.load_dataset(DATASET, None)

    train, test = benchmark.stable_split(
        records,
        test_fraction=TEST_FRACTION,
        seed=SEED,
    )

    cases = benchmark.make_cases(test)

    print(f"[E2] Records: {len(records)}")
    print(f"[E2] Training scenarios: {len(train)}")
    print(f"[E2] Test scenarios: {len(test)}")
    print(f"[E2] Test queries: {len(cases)}")

    repository = benchmark.build_repository(train, test)

    # This index is built directly from the complete training records.
    # It is independent of GraphNeighborhoodRetriever's edge limit.
    support_index = GlobalPathSupport(train, test)

    builder = BuildEvidenceContext()
    reasoner = GraphReasoner(max_hops=MAX_HOPS)

    conditions = {}

    for threshold in SUPPORT_THRESHOLDS:
        conditions[f"minimum_path_support_{threshold}"] = (
            run_condition(
                threshold=threshold,
                repository=repository,
                support_index=support_index,
                cases=cases,
                builder=builder,
                reasoner=reasoner,
            )
        )

    dataset_path = benchmark.DATASET_PATHS[DATASET]

    dataset_hash = hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()

    output = {
        "metadata": {
            "experiment": "E2",
            "title": "Global complete-path evidence consensus",
            "protocol_version": "E2-3.0",
            "research_question": (
                "How does the number of distinct eligible training "
                "scenarios supporting a complete typed path affect "
                "missing-link recovery?"
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
            "support_thresholds": list(SUPPORT_THRESHOLDS),
            "test_scenario_ids": sorted(
                record.scenario_id for record in test
            ),
            "case_ids": [case.case_id for case in cases],
            "path_definition": (
                "Cause --causes_deviation--> Deviation "
                "--leads_to_consequence--> Consequence"
            ),
            "support_definition": (
                "Number of distinct eligible training scenario IDs "
                "containing both typed edges of the same complete path."
            ),
            "support_scope": (
                "All training records in the target plant unit; "
                "equipment is used when plant unit is unavailable."
            ),
            "retrieval_scope": (
                "GraphNeighborhoodRetriever; its bounded retrieved "
                "edges are filtered using independently computed "
                "global support counts."
            ),
            "threshold_1": (
                "Unmodified graph-reasoning baseline."
            ),
            "thresholds_above_1": (
                "Retain retrieved source edges only when they belong "
                "to a complete path meeting the global support threshold."
            ),
            "llm_enabled": False,
            "label_warning": (
                "Reconstructed row-level associations are silver labels, "
                "not expert-validated causal ground truth."
            ),
            "independence_warning": (
                "Distinct scenarios may share source templates and "
                "underlying assumptions. Scenario counts do not establish "
                "statistical independence or physical confirmation."
            ),
        },
        "conditions": conditions,
    }

    output_path = (
        ROOT / "results" / "e2" / "global_path_consensus.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== E2 GLOBAL PATH CONSENSUS ==========")

    for name, condition in conditions.items():
        metrics = condition["summary"]

        print(
            f"{name:28s} | "
            f"F1={metrics['micro_f1']:.4f} | "
            f"P={metrics['micro_precision']:.4f} | "
            f"R={metrics['micro_recall']:.4f} | "
            f"Coverage={metrics['coverage']:.4f} | "
            f"Abstention={metrics['abstention_rate']:.4f}"
        )

    print(f"\n[E2] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
