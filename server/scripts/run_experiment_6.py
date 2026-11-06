"""E6: Robustness to scenario-level split variation.

Run from the server directory:
    uv run python scripts/run_experiment_6.py

This evaluates GraphReasoner on repeated deterministic 80/20 scenario-level
splits. It does not pool overlapping test queries across seeds as independent
observations. Labels are reconstructed worksheet co-associations, not
expert-validated physical causal ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import scripts.run_benchmark as benchmark
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from scripts.run_experiment_4 import run_condition

DATASET = "t301-reconstructed"
DEFAULT_SEEDS = tuple(range(42, 52))
TEST_FRACTION = 0.2
TOP_K = 10
MAX_HOPS = 4
REASONER_MODE = "both"

METRIC_KEYS = (
    "micro_precision", "micro_recall", "micro_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "coverage", "abstention_rate",
)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def scenario_metrics(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = {}
    for case in cases:
        counts = grouped.setdefault(case["scenario_id"], {
            "tp": 0, "fp": 0, "fn": 0, "queries": 0, "covered": 0,
        })
        gold = {
            benchmark.normalize(str(v))
            for v in case.get("gold_values", []) if str(v).strip()
        }
        predicted = {
            benchmark.normalize(str(v))
            for v in case.get("normalized_prediction_values", [])
            if str(v).strip()
        }
        counts["tp"] += len(gold & predicted)
        counts["fp"] += len(predicted - gold)
        counts["fn"] += len(gold - predicted)
        counts["queries"] += 1
        counts["covered"] += int(bool(predicted))

    output: dict[str, dict[str, Any]] = {}
    for scenario_id, counts in sorted(grouped.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        output[scenario_id] = {
            **counts, "precision": p, "recall": r, "f1": f1,
            "coverage": counts["covered"] / counts["queries"]
            if counts["queries"] else 0.0,
        }
    return output


def summarize_seed_metrics(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric in METRIC_KEYS:
        values = [
            float(run["metrics"][metric])
            for run in runs if run["metrics"].get(metric) is not None
        ]
        summary[metric] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else (
                0.0 if values else None
            ),
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def run_seed(
    *, seed: int, records, test_fraction: float, top_k: int, max_hops: int,
) -> dict[str, Any]:
    train, test = benchmark.stable_split(
        records, test_fraction=test_fraction, seed=seed,
    )
    cases = benchmark.make_cases(test)
    if not cases:
        raise RuntimeError(f"Seed {seed} produced no eligible held-out queries.")

    train_scenarios = sorted({r.scenario_id for r in train})
    test_scenarios = sorted({r.scenario_id for r in test})
    overlap = set(train_scenarios) & set(test_scenarios)
    if overlap:
        raise RuntimeError(
            f"Scenario leakage detected for seed {seed}: {sorted(overlap)}"
        )

    repository = benchmark.build_repository(train, test)
    result = run_condition(
        mode=REASONER_MODE,
        repository=repository,
        cases=cases,
        builder=BuildEvidenceContext(),
        top_k=top_k,
        max_hops=max_hops,
    )
    saved_cases = result["cases"]
    return {
        "seed": seed,
        "split": {
            "training_scenario_count": len(train_scenarios),
            "test_scenario_count": len(test_scenarios),
            "training_scenarios": train_scenarios,
            "test_scenarios": test_scenarios,
            "train_test_scenario_overlap": sorted(overlap),
            "test_query_count": len(cases),
        },
        "metrics": result["summary"],
        "scenario_metrics": scenario_metrics(saved_cases),
        "cases": saved_cases,
        "path_diagnostics": result.get("aggregate_path_diagnostics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E6 repeated scenario-level split robustness evaluation."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Deterministic split seeds (default: 42 through 51).",
    )
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-hops", type=int, default=MAX_HOPS)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional output path; default is a timestamped file under results/e6/.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing the explicitly supplied --output path.",
    )
    args = parser.parse_args()

    if not args.seeds:
        parser.error("At least one --seeds value is required.")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates.")
    if not 0.0 < args.test_fraction < 1.0:
        parser.error("--test-fraction must be between 0 and 1.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.max_hops < 2:
        parser.error("--max-hops must be at least 2.")

    root = benchmark.ROOT
    dataset_path = benchmark.DATASET_PATHS[DATASET]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        root / "results" / "e6" / f"scenario_split_robustness_{timestamp}.json"
        if args.output is None else args.output.expanduser().resolve()
    )
    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path}. Choose another --output "
            "or pass --overwrite only if replacement is intentional."
        )

    print(f"[E6] Repository root: {root}")
    print(f"[E6] Dataset: {DATASET}")
    records = benchmark.load_dataset(DATASET, None)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    scenario_count = len({r.scenario_id for r in records})
    print(
        f"[E6] Records={len(records)} | scenarios={scenario_count} | "
        f"seeds={args.seeds} | test_fraction={args.test_fraction:.3f}"
    )
    print(
        f"[E6] Method=GraphReasoner(mode={REASONER_MODE}) | "
        f"top_k={args.top_k} | max_hops={args.max_hops} | LLM=disabled"
    )

    runs: list[dict[str, Any]] = []
    split_signatures: dict[tuple[str, ...], list[int]] = {}
    for index, seed in enumerate(args.seeds, start=1):
        print(f"\n[E6] Split {index}/{len(args.seeds)} | seed={seed}")
        run = run_seed(
            seed=seed, records=records, test_fraction=args.test_fraction,
            top_k=args.top_k, max_hops=args.max_hops,
        )
        signature = tuple(run["split"]["test_scenarios"])
        split_signatures.setdefault(signature, []).append(seed)
        runs.append(run)
        metrics = run["metrics"]
        print(
            f"[E6] seed={seed} | queries={run['split']['test_query_count']} | "
            f"P={metrics['micro_precision']:.4f} "
            f"R={metrics['micro_recall']:.4f} "
            f"F1={metrics['micro_f1']:.4f} "
            f"coverage={metrics['coverage']:.4f}"
        )

    unique_test_scenarios = sorted({
        scenario_id for run in runs
        for scenario_id in run["split"]["test_scenarios"]
    })
    output = {
        "metadata": {
            "experiment": "E6",
            "title": "Robustness to scenario-level split variation",
            "protocol_version": "E6-1.0",
            "research_question": (
                "How sensitive is graph-based HAZOP relationship recovery "
                "to the particular scenario-level train/test split?"
            ),
            "dataset": DATASET,
            "dataset_path": str(dataset_path.relative_to(root)),
            "dataset_sha256": dataset_hash,
            "record_count": len(records),
            "unique_scenario_count": scenario_count,
            "seeds": list(args.seeds),
            "test_fraction": args.test_fraction,
            "top_k": args.top_k,
            "max_hops": args.max_hops,
            "reasoner": "GraphReasoner",
            "graph_reasoner_mode": REASONER_MODE,
            "llm_enabled": False,
            "split_protocol": (
                "Each seed independently applies benchmark.stable_split(), "
                "which partitions whole scenario IDs. All rows belonging to a "
                "scenario remain in one partition. Held-out scenario labels "
                "are stripped by benchmark.build_repository()."
            ),
            "comparison_unit": (
                "Per-seed split metrics. Test sets can overlap across seeds; "
                "queries across seeds are not pooled or treated as independent."
            ),
            "gold_label_semantics": (
                "Reconstructed worksheet row co-associations; not "
                "independently expert-validated physical causal ground truth."
            ),
            "unique_test_scenario_coverage": {
                "count": len(unique_test_scenarios),
                "scenario_ids": unique_test_scenarios,
            },
            "duplicate_split_groups": [
                {"test_scenarios": list(signature), "seeds": seeds}
                for signature, seeds in sorted(split_signatures.items())
                if len(seeds) > 1
            ],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(root),
        },
        "across_seed_summary": summarize_seed_metrics(runs),
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("\n[E6] ACROSS-SEED SUMMARY")
    print(
        f"{'Metric':<20} {'Mean':>10} {'Std':>10} "
        f"{'Median':>10} {'Min':>10} {'Max':>10}"
    )
    for metric, values in output["across_seed_summary"].items():
        if values["mean"] is None:
            continue
        print(
            f"{metric:<20} {values['mean']:>10.4f} "
            f"{values['std']:>10.4f} {values['median']:>10.4f} "
            f"{values['min']:>10.4f} {values['max']:>10.4f}"
        )
    print(
        f"\n[E6] Unique held-out scenarios across all runs: "
        f"{len(unique_test_scenarios)}/{scenario_count}"
    )
    print(f"[E6] Results written to: {output_path}")


if __name__ == "__main__":
    main()
