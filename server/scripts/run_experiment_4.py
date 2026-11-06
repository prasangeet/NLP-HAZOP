"""E4: Controlled evaluation of typed multi-hop graph reasoning.

Research question:
How does the permitted typed path length affect recovery of held-out
cause/consequence worksheet associations?

Conditions:
- direct:       cause -> deviation -> consequence (2 edges)
- intermediate: cause -> deviation -> effect -> consequence (3 edges)
- both:         either pattern, retaining the reasoner's shortest-path choice

Protocol:
- Reuses the frozen E1 dataset, scenario split, and query construction.
- Uses only t301-reconstructed records (no synthetic continuation rows).
- Uses the same graph-neighborhood retrieval and recovery pipeline in every
  condition. Only GraphReasoner's association_path_mode changes.
- Does not invoke an LLM or alter E1/E2/E3 outputs.

Default output:
    results/e4/multi_hop_reasoning.json

The labels are reconstructed worksheet co-associations, not expert-validated
physical causal ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
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
    "direct": "Two-edge cause -> deviation -> consequence path only.",
    "intermediate": (
        "Three-edge cause -> deviation -> effect -> consequence path only."
    ),
    "both": (
        "Both supported path patterns; shortest path selected per endpoint."
    ),
}


def normalize(value: Any) -> str:
    """Use the benchmark's canonical entity normalization."""
    return benchmark.normalize(str(value))


def display_prediction(value: Any) -> str:
    """Extract a displayable endpoint from benchmark prediction values."""
    if value is None:
        return ""
    if isinstance(value, dict):
        if value.get("value") is not None:
            return display_prediction(value["value"])
        for key in ("candidate", "entity", "endpoint", "object", "label", "text"):
            if value.get(key) is not None:
                return display_prediction(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def serialize_path(path) -> dict[str, Any]:
    """Serialize the actual internal path returned by GraphReasoner."""
    return {
        "nodes": list(path.display_nodes),
        "typed_nodes": [
            {"entity_type": node_type.value, "value": value}
            for node_type, value in path.nodes
        ],
        "endpoint": path.display_endpoint,
        "normalized_endpoint": normalize(path.display_endpoint),
        "hop_count": len(path.edges),
        "edges": [
            {
                "subject": edge.subject,
                "predicate": edge.predicate,
                "object": edge.object,
                "scenario_id": getattr(edge, "scenario_id", None),
                "dataset_id": getattr(edge, "dataset_id", None),
                "source_id": getattr(edge, "source_id", None),
                "source_location": getattr(edge, "source_location", None),
                "origin": getattr(
                    getattr(edge, "origin", None), "value",
                    str(getattr(edge, "origin", "")),
                ),
                "evidence": getattr(edge, "evidence", None),
            }
            for edge in path.edges
        ],
    }


class AuditedGraphReasoner(GraphReasoner):
    """GraphReasoner that records discovered and selected paths per query."""

    def __init__(self, *, max_hops: int, association_path_mode: str):
        super().__init__(
            max_hops=max_hops,
            association_path_mode=association_path_mode,
        )
        self.reset_audit()

    def reset_audit(self) -> None:
        self.last_candidate_paths: list[dict[str, Any]] = []
        self.last_selected_paths: list[dict[str, Any]] = []

    def _find_paths(self, context, direction):
        paths = super()._find_paths(context, direction)
        self.last_candidate_paths = [serialize_path(path) for path in paths]
        return paths

    def _build_path_relation(self, path, *, context, dataset_id):
        relation = super()._build_path_relation(
            path,
            context=context,
            dataset_id=dataset_id,
        )
        self.last_selected_paths.append({
            "path": serialize_path(path),
            "relation_constructed": relation is not None,
            "constructed_relation": (
                {
                    "subject": relation.subject,
                    "predicate": relation.predicate,
                    "object": relation.object,
                    "origin": getattr(
                        getattr(relation, "origin", None), "value",
                        str(getattr(relation, "origin", "")),
                    ),
                    "source_id": relation.source_id,
                    "evidence": relation.evidence,
                }
                if relation is not None else None
            ),
        })
        return relation


def run_condition(
    *,
    mode: str,
    repository,
    cases,
    builder: BuildEvidenceContext,
    top_k: int,
    max_hops: int,
) -> dict[str, Any]:
    """Run every frozen test query with one path policy."""
    local_retriever = ScenarioLocalRetriever(repository)
    graph_retriever = GraphNeighborhoodRetriever(
        local_retriever,
        repository,
    )
    use_case = RecoverMissingLink(graph_retriever)
    reasoner = AuditedGraphReasoner(
        max_hops=max_hops,
        association_path_mode=mode,
    )

    evaluated_cases: list[dict[str, Any]] = []
    print(f"\n[E4] Condition={mode} | {CONDITIONS[mode]}")

    for index, case in enumerate(cases, start=1):
        request = RecoveryRequest(
            scenario_id=case.scenario_id,
            predicate="co_listed_with_consequence",
            known_endpoint=case.known_endpoint,
            missing_endpoint=MissingEndpoint.OBJECT,
            limit=top_k,
            dataset_id=case.dataset_id,
        )

        reasoner.reset_audit()
        predictions, latency_ms, status, diagnostics = benchmark._run_recovery(
            use_case,
            builder,
            request,
            reasoner=reasoner,
        )

        prediction_values = list(dict.fromkeys(
            display_prediction(item)
            for item in predictions
            if display_prediction(item)
        ))
        prediction_set = {normalize(value) for value in prediction_values}

        candidate_paths = list(reasoner.last_candidate_paths)
        selected_paths = list(reasoner.last_selected_paths)
        candidate_hops = Counter(
            str(path["hop_count"]) for path in candidate_paths
        )
        selected_hops = Counter(
            str(item["path"]["hop_count"]) for item in selected_paths
        )

        gold_diagnostics = []
        for gold in case.gold_values:
            key = normalize(gold)
            candidate_matches = [
                path for path in candidate_paths
                if path["normalized_endpoint"] == key
            ]
            selected_matches = [
                item for item in selected_paths
                if item["path"]["normalized_endpoint"] == key
            ]
            gold_diagnostics.append({
                "gold_consequence": gold,
                "predicted": key in prediction_set,
                "candidate_path_count": len(candidate_matches),
                "selected_path_count": len(selected_matches),
                "candidate_paths": candidate_matches,
                "selected_paths": selected_matches,
                "failure_stage": (
                    "prediction_emitted" if key in prediction_set else
                    "path_selected_but_not_emitted" if selected_matches else
                    "candidate_path_found_but_not_selected" if candidate_matches else
                    "no_candidate_path_found"
                ),
            })

        evaluated_cases.append({
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "known_endpoint": case.known_endpoint,
            "gold_values": list(case.gold_values),
            "predictions": predictions,
            "normalized_prediction_values": prediction_values,
            "latency_ms": latency_ms,
            "status": status,
            "diagnostics": diagnostics,
            "path_audit": {
                "candidate_path_count": len(candidate_paths),
                "selected_path_count": len(selected_paths),
                "candidate_hop_count_distribution": dict(sorted(candidate_hops.items())),
                "selected_hop_count_distribution": dict(sorted(selected_hops.items())),
                "candidate_paths": candidate_paths,
                "selected_paths": selected_paths,
                "gold_diagnostics": gold_diagnostics,
            },
        })

        print(
            f"[E4] {index}/{len(cases)} {case.case_id} | "
            f"candidate_paths={len(candidate_paths)} | "
            f"selected_paths={len(selected_paths)} | "
            f"predictions={len(predictions)} | status={status}"
        )

    return {
        "mode": mode,
        "mode_definition": CONDITIONS[mode],
        "summary": benchmark.score_predictions(evaluated_cases),
        "aggregate_path_diagnostics": aggregate_path_diagnostics(evaluated_cases),
        "cases": evaluated_cases,
    }


def aggregate_path_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate path discovery, selection, and gold failure stages."""
    candidate_hops: Counter[str] = Counter()
    selected_hops: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    gold_total = 0
    gold_predicted = 0
    gold_with_candidate_path = 0
    gold_with_selected_path = 0

    for case in cases:
        audit = case["path_audit"]
        candidate_hops.update(audit["candidate_hop_count_distribution"])
        selected_hops.update(audit["selected_hop_count_distribution"])
        for item in audit["gold_diagnostics"]:
            gold_total += 1
            gold_predicted += bool(item["predicted"])
            gold_with_candidate_path += item["candidate_path_count"] > 0
            gold_with_selected_path += item["selected_path_count"] > 0
            failure_stages[item["failure_stage"]] += 1

    return {
        "candidate_path_hop_count_distribution": dict(sorted(candidate_hops.items())),
        "selected_path_hop_count_distribution": dict(sorted(selected_hops.items())),
        "gold_consequence_count": gold_total,
        "gold_consequences_predicted": gold_predicted,
        "gold_consequences_with_candidate_path": gold_with_candidate_path,
        "gold_consequences_with_selected_path": gold_with_selected_path,
        "gold_failure_stage_counts": dict(sorted(failure_stages.items())),
    }


def build_mode_comparisons(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare direct/intermediate outputs against the legacy both-mode output."""
    baseline_cases = {
        case["case_id"]: case
        for case in conditions["both"]["cases"]
    }
    comparisons: dict[str, Any] = {}

    for mode in ("direct", "intermediate"):
        rows = []
        other_cases = {
            case["case_id"]: case
            for case in conditions[mode]["cases"]
        }
        for case_id, baseline in baseline_cases.items():
            other = other_cases[case_id]
            gold = {normalize(value): value for value in baseline["gold_values"]}
            both_predictions = {
                normalize(display_prediction(value))
                for value in baseline["predictions"]
            }
            mode_predictions = {
                normalize(display_prediction(value))
                for value in other["predictions"]
            }
            baseline_gold = set(gold) & both_predictions
            mode_gold = set(gold) & mode_predictions
            rows.append({
                "case_id": case_id,
                "scenario_id": baseline["scenario_id"],
                "gold_lost_vs_both": [gold[key] for key in sorted(baseline_gold - mode_gold)],
                "gold_retained_vs_both": [gold[key] for key in sorted(baseline_gold & mode_gold)],
                "gold_newly_recovered_vs_both": [gold[key] for key in sorted(mode_gold - baseline_gold)],
                "both_predictions": baseline["normalized_prediction_values"],
                f"{mode}_predictions": other["normalized_prediction_values"],
            })
        comparisons[mode] = rows

    return comparisons


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E4 controlled multi-hop graph reasoning evaluation."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-hops", type=int, default=MAX_HOPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=(benchmark.ROOT / "results" / "e4" / "multi_hop_reasoning.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the specified output file.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.max_hops < 3:
        parser.error("--max-hops must be at least 3 for the intermediate path")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path}. Use --overwrite only if "
            "you intentionally want to replace it, or choose another --output."
        )

    root = benchmark.ROOT
    dataset_path = benchmark.DATASET_PATHS[DATASET]
    print(f"[E4] Repository root: {root}")
    print(f"[E4] Dataset: {DATASET}")
    print("[E4] Loading reconstructed records...")

    records = benchmark.load_dataset(DATASET, None)
    train, test = benchmark.stable_split(
        records,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    cases = benchmark.make_cases(test)
    if not cases:
        raise RuntimeError("The held-out split produced no benchmark queries.")

    print(
        f"[E4] Records={len(records)} | train_scenarios={len(train)} | "
        f"test_scenarios={len(test)} | test_queries={len(cases)}"
    )

    repository = benchmark.build_repository(train, test)
    builder = BuildEvidenceContext()

    conditions = {
        mode: run_condition(
            mode=mode,
            repository=repository,
            cases=cases,
            builder=builder,
            top_k=args.top_k,
            max_hops=args.max_hops,
        )
        for mode in ("direct", "intermediate", "both")
    }

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    output = {
        "metadata": {
            "experiment": "E4",
            "title": "Controlled evaluation of typed multi-hop graph reasoning",
            "protocol_version": "E4-1.0",
            "research_question": (
                "How does the permitted typed path length affect recovery "
                "of held-out cause/consequence worksheet associations?"
            ),
            "dataset": DATASET,
            "dataset_path": str(dataset_path.relative_to(root)),
            "dataset_sha256": dataset_hash,
            "record_count": len(records),
            "training_scenario_count": len(train),
            "test_scenario_count": len(test),
            "test_query_count": len(cases),
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "top_k": args.top_k,
            "max_hops": args.max_hops,
            "llm_enabled": False,
            "conditions": CONDITIONS,
            "split_protocol": (
                "Same deterministic scenario-level split procedure as E1; "
                "all rows belonging to a scenario remain in one partition."
            ),
            "gold_label_semantics": (
                "Reconstructed worksheet row co-associations; not "
                "independently expert-validated physical causal ground truth."
            ),
            "path_selection_semantics": (
                "GraphReasoner.reason_all selects the shortest valid path "
                "per endpoint. Therefore, in both mode a direct path takes "
                "precedence over a three-edge path to the same endpoint."
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(root),
        },
        "conditions": conditions,
        "comparisons_against_both": build_mode_comparisons(conditions),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[E4] SUMMARY")
    print(
        f"{'Mode':<16} {'P':>8} {'R':>8} {'F1':>8} "
        f"{'Coverage':>10} {'Abstain':>10} {'Cand paths':>12}"
    )
    for mode in ("direct", "intermediate", "both"):
        result = conditions[mode]
        metrics = result["summary"]
        paths = result["aggregate_path_diagnostics"]
        print(
            f"{mode:<16} "
            f"{metrics['micro_precision']:>8.4f} "
            f"{metrics['micro_recall']:>8.4f} "
            f"{metrics['micro_f1']:>8.4f} "
            f"{metrics['coverage']:>10.4f} "
            f"{metrics['abstention_rate']:>10.4f} "
            f"{sum(int(v) for v in paths['candidate_path_hop_count_distribution'].values()):>12}"
        )

    print(f"\n[E4] Results written to: {output_path}")


if __name__ == "__main__":
    main()
