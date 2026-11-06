"""E5 v2: Provenance-coherence gating and selective prediction risk.

This version replaces metadata-completeness thresholds with evidence-scope
policies. It compares:
  1. ungated proposals,
  2. paths whose edges come from one scenario,
  3. paths whose edges come from the same reconstructed source row.

A same-row path is stronger *documentary coherence* than a path assembled
across scenarios. It is NOT proof of physical causality or expert validation.

The original E5-1.0 output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import scripts.run_benchmark as benchmark
from scripts.run_experiment_4 import AuditedGraphReasoner, display_prediction

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

POLICIES = (
    "ungated",
    "same_scenario",
    "same_source_row",
)


def normalize(value: Any) -> str:
    return benchmark.normalize(str(value))


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


def source_row_key(source_id: Any) -> str | None:
    """Return dataset:scenario:row prefix from a canonical source identifier."""
    if not isinstance(source_id, str) or not source_id.strip():
        return None
    parts = source_id.strip().split(":")
    # Expected form:
    # dataset-id:scenario-id:row-N:cause|effect|consequence:index
    if len(parts) < 5 or not parts[1] or not parts[2].startswith("row-"):
        return None
    return ":".join(parts[:3])


def inspect_path(path: dict[str, Any] | None) -> dict[str, Any]:
    """Describe provenance completeness and coherence without conflating them."""
    if not path:
        return {
            "path_present": False,
            "edge_count": 0,
            "all_edges_grounded": False,
            "scenario_coherent": False,
            "source_row_coherent": False,
            "scenario_ids": [],
            "source_row_keys": [],
            "origins": [],
            "edge_traceability": [],
        }

    edges = path.get("edges") or []
    scenario_ids = {
        str(edge.get("scenario_id")).strip()
        for edge in edges
        if edge.get("scenario_id") not in (None, "")
    }
    row_keys = {
        key for edge in edges
        if (key := source_row_key(edge.get("source_id"))) is not None
    }
    origins = {
        str(edge.get("origin")).strip().lower()
        for edge in edges
        if edge.get("origin") not in (None, "")
    }

    edge_rows = []
    for edge in edges:
        scenario_present = edge.get("scenario_id") not in (None, "")
        source_present = source_row_key(edge.get("source_id")) is not None
        evidence_present = bool(str(edge.get("evidence") or "").strip())
        edge_rows.append({
            "subject": edge.get("subject"),
            "predicate": edge.get("predicate"),
            "object": edge.get("object"),
            "scenario_id": edge.get("scenario_id"),
            "source_id": edge.get("source_id"),
            "source_row_key": source_row_key(edge.get("source_id")),
            "source_location": edge.get("source_location"),
            "origin": edge.get("origin"),
            "has_scenario_id": scenario_present,
            "has_valid_source_id": source_present,
            "has_evidence_text": evidence_present,
            "grounded": scenario_present and source_present and evidence_present,
        })

    all_grounded = bool(edges) and all(item["grounded"] for item in edge_rows)
    scenario_coherent = (
        bool(edges)
        and all(item["has_scenario_id"] for item in edge_rows)
        and len(scenario_ids) == 1
    )
    source_row_coherent = (
        bool(edges)
        and all(item["grounded"] for item in edge_rows)
        and len(row_keys) == 1
    )

    return {
        "path_present": bool(edges),
        "edge_count": len(edges),
        "all_edges_grounded": all_grounded,
        "scenario_coherent": scenario_coherent,
        "source_row_coherent": source_row_coherent,
        "scenario_ids": sorted(scenario_ids),
        "source_row_keys": sorted(row_keys),
        "origins": sorted(origins),
        "edge_traceability": edge_rows,
    }


def collect_base_cases(*, repository, cases, top_k: int, max_hops: int) -> list[dict[str, Any]]:
    """Run graph reasoning once; evaluate multiple gates over identical proposals."""
    local_retriever = ScenarioLocalRetriever(repository)
    graph_retriever = GraphNeighborhoodRetriever(local_retriever, repository)
    use_case = RecoverMissingLink(graph_retriever)
    reasoner = AuditedGraphReasoner(
        max_hops=max_hops,
        association_path_mode="both",
    )
    builder = BuildEvidenceContext()
    output: list[dict[str, Any]] = []

    print("\n[E5-v2] Collecting ungated graph proposals and path provenance...")
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

        selected_by_endpoint: dict[str, dict[str, Any]] = {}
        for item in reasoner.last_selected_paths:
            path = item.get("path") or {}
            endpoint = path.get("normalized_endpoint")
            if endpoint and item.get("relation_constructed"):
                selected_by_endpoint.setdefault(endpoint, path)

        proposal_rows = []
        seen: set[str] = set()
        for raw_prediction in predictions:
            value = display_prediction(raw_prediction)
            key = normalize(value) if value else ""
            if not key or key in seen:
                continue
            seen.add(key)
            path = selected_by_endpoint.get(key)
            provenance = inspect_path(path)
            proposal_rows.append({
                "value": value,
                "normalized_value": key,
                "raw_prediction": raw_prediction,
                "selected_path": path,
                "provenance": provenance,
                "evidence_scope": (
                    "same_source_row"
                    if provenance["source_row_coherent"]
                    else "same_scenario"
                    if provenance["scenario_coherent"]
                    else "cross_scenario_or_unresolved"
                ),
            })

        output.append({
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "known_endpoint": case.known_endpoint,
            "gold_values": list(case.gold_values),
            "gold_normalized": sorted({normalize(value) for value in case.gold_values}),
            "proposals": proposal_rows,
            "latency_ms": latency_ms,
            "base_status": status,
            "base_diagnostics": diagnostics,
            "base_candidate_path_count": len(reasoner.last_candidate_paths),
            "base_selected_path_count": len(reasoner.last_selected_paths),
        })
        print(
            f"[E5-v2] {index}/{len(cases)} {case.case_id} | "
            f"proposals={len(proposal_rows)} | status={status}"
        )

    return output


def accepts(proposal: dict[str, Any], policy: str) -> bool:
    provenance = proposal["provenance"]
    if policy == "ungated":
        return True
    if policy == "same_scenario":
        return bool(
            provenance["all_edges_grounded"]
            and provenance["scenario_coherent"]
        )
    if policy == "same_source_row":
        return bool(provenance["source_row_coherent"])
    raise ValueError(f"Unknown E5 policy: {policy}")


def score_policy(base_cases: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    tp = fp = fn = accepted_count = 0
    unsupported_count = scenario_coherent_count = row_coherent_count = 0
    covered_cases = 0
    per_case = []
    scenario_scores: dict[str, dict[str, int]] = {}

    for case in base_cases:
        gold = set(case["gold_normalized"])
        accepted = [
            proposal for proposal in case["proposals"]
            if accepts(proposal, policy)
        ]
        rejected = [
            proposal for proposal in case["proposals"]
            if not accepts(proposal, policy)
        ]
        predicted = {proposal["normalized_value"] for proposal in accepted}
        case_tp = len(predicted & gold)
        case_fp = len(predicted - gold)
        case_fn = len(gold - predicted)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        accepted_count += len(accepted)
        unsupported_count += sum(
            not proposal["provenance"]["all_edges_grounded"]
            for proposal in accepted
        )
        scenario_coherent_count += sum(
            proposal["provenance"]["scenario_coherent"]
            for proposal in accepted
        )
        row_coherent_count += sum(
            proposal["provenance"]["source_row_coherent"]
            for proposal in accepted
        )
        if accepted:
            covered_cases += 1

        p = case_tp / (case_tp + case_fp) if case_tp + case_fp else 0.0
        r = case_tp / (case_tp + case_fn) if case_tp + case_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_case.append({
            "case_id": case["case_id"],
            "scenario_id": case["scenario_id"],
            "gold_values": case["gold_values"],
            "accepted_predictions": [item["value"] for item in accepted],
            "rejected_predictions": [item["value"] for item in rejected],
            "accepted_proposal_count": len(accepted),
            "rejected_proposal_count": len(rejected),
            "true_positives": case_tp,
            "false_positives": case_fp,
            "false_negatives": case_fn,
            "precision": p,
            "recall": r,
            "f1": f1,
            "query_abstained": not bool(accepted),
            "accepted_evidence": [
                {
                    "prediction": item["value"],
                    "evidence_scope": item["evidence_scope"],
                    "all_edges_grounded": item["provenance"]["all_edges_grounded"],
                    "scenario_coherent": item["provenance"]["scenario_coherent"],
                    "source_row_coherent": item["provenance"]["source_row_coherent"],
                    "scenario_ids": item["provenance"]["scenario_ids"],
                    "source_row_keys": item["provenance"]["source_row_keys"],
                }
                for item in accepted
            ],
        })
        scenario = scenario_scores.setdefault(
            case["scenario_id"],
            {"tp": 0, "fp": 0, "fn": 0, "queries": 0, "covered": 0},
        )
        scenario["tp"] += case_tp
        scenario["fp"] += case_fp
        scenario["fn"] += case_fn
        scenario["queries"] += 1
        scenario["covered"] += int(bool(accepted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_precision = sum(row["precision"] for row in per_case) / len(per_case) if per_case else 0.0
    macro_recall = sum(row["recall"] for row in per_case) / len(per_case) if per_case else 0.0
    macro_f1 = sum(row["f1"] for row in per_case) / len(per_case) if per_case else 0.0
    coverage = covered_cases / len(base_cases) if base_cases else 0.0
    selective_risk = 1.0 - precision if accepted_count else None

    scenario_metrics = {}
    for scenario_id, counts in sorted(scenario_scores.items()):
        s_p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        s_r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        s_f1 = 2 * s_p * s_r / (s_p + s_r) if s_p + s_r else 0.0
        scenario_metrics[scenario_id] = {
            **counts,
            "precision": s_p,
            "recall": s_r,
            "f1": s_f1,
            "coverage": counts["covered"] / counts["queries"] if counts["queries"] else 0.0,
        }

    return {
        "policy": policy,
        "policy_semantics": {
            "ungated": "Accept every GraphReasoner proposal with a selected path.",
            "same_scenario": "Accept only when every path edge is grounded and all edges originate from one scenario.",
            "same_source_row": "Accept only when every path edge is grounded and all edge source IDs identify the same dataset/scenario/row.",
        }[policy],
        "counts": {
            "queries": len(base_cases),
            "covered_queries": covered_cases,
            "abstained_queries": len(base_cases) - covered_cases,
            "accepted_proposals": accepted_count,
            "rejected_proposals": sum(len(case["proposals"]) for case in base_cases) - accepted_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "unsupported_accepted_proposals": unsupported_count,
            "same_scenario_accepted_proposals": scenario_coherent_count,
            "same_source_row_accepted_proposals": row_coherent_count,
        },
        "metrics": {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "coverage": coverage,
            "abstention_rate": 1.0 - coverage,
            "selective_precision": precision if accepted_count else None,
            "selective_risk": selective_risk,
            "unsupported_proposal_rate": unsupported_count / accepted_count if accepted_count else None,
            "same_scenario_rate": scenario_coherent_count / accepted_count if accepted_count else None,
            "same_source_row_rate": row_coherent_count / accepted_count if accepted_count else None,
        },
        "per_scenario": scenario_metrics,
        "per_query": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run E5-v2 provenance-coherence abstention evaluation."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-hops", type=int, default=MAX_HOPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            benchmark.ROOT
            / "results"
            / "e5"
            / "abstention_selective_risk_v2.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.max_hops < 2:
        parser.error("--max-hops must be at least 2")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path}. Choose another --output "
            "or pass --overwrite only if replacement is intentional."
        )

    root = benchmark.ROOT
    dataset_path = benchmark.DATASET_PATHS[DATASET]
    print(f"[E5-v2] Repository root: {root}")
    print(f"[E5-v2] Dataset: {DATASET}")
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
        f"[E5-v2] Records={len(records)} | train_scenarios={len(train)} | "
        f"test_scenarios={len(test)} | test_queries={len(cases)}"
    )
    repository = benchmark.build_repository(train, test)
    base_cases = collect_base_cases(
        repository=repository,
        cases=cases,
        top_k=args.top_k,
        max_hops=args.max_hops,
    )

    conditions = {
        policy: score_policy(base_cases, policy)
        for policy in POLICIES
    }

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    output = {
        "metadata": {
            "experiment": "E5",
            "title": "Provenance-coherence gating and selective prediction risk",
            "protocol_version": "E5-2.0",
            "research_question": (
                "Does requiring scenario-coherent or source-row-coherent typed "
                "paths improve accepted-proposal reliability, and what coverage "
                "is sacrificed?"
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
            "graph_reasoner_mode": "both",
            "split_protocol": (
                "Same deterministic scenario-level split procedure as E1; "
                "all records from a scenario remain in one partition."
            ),
            "proposal_source": "GraphReasoner selected typed paths; no LLM proposals.",
            "policy_definitions": {
                "ungated": "Accept all graph proposals.",
                "same_scenario": "Require every edge to have scenario/source/evidence provenance and require all edges to originate in one scenario.",
                "same_source_row": "Require every edge to be grounded and all edge source IDs to resolve to one source row.",
            },
            "provenance_interpretation": (
                "Scenario/row coherence measures documentary consistency of "
                "the reconstructed records. It is not calibrated confidence, "
                "statistical independence, expert validation, or physical causality."
            ),
            "unsupported_definition": (
                "A proposal is unsupported when its selected path is absent, "
                "or any path edge lacks a scenario ID, parseable source-row ID, "
                "or non-empty evidence text."
            ),
            "selective_risk_definition": "1 - micro precision among accepted proposals.",
            "coverage_definition": "Fraction of test queries with at least one accepted proposal.",
            "gold_label_semantics": (
                "Reconstructed worksheet row co-associations; not independently "
                "expert-validated physical causal ground truth."
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(root),
        },
        "conditions": conditions,
        "base_cases_with_provenance": base_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[E5-v2] SELECTIVE PREDICTION SUMMARY")
    print(
        f"{'Policy':<18} {'Psel':>8} {'Risk':>8} {'F1':>8} "
        f"{'Coverage':>10} {'Abstain':>10} {'Accepted':>10}"
    )
    for policy, result in conditions.items():
        metrics = result["metrics"]
        p = metrics["selective_precision"]
        risk = metrics["selective_risk"]
        print(
            f"{policy:<18} "
            f"{p if p is not None else float('nan'):>8.4f} "
            f"{risk if risk is not None else float('nan'):>8.4f} "
            f"{metrics['micro_f1']:>8.4f} "
            f"{metrics['coverage']:>10.4f} "
            f"{metrics['abstention_rate']:>10.4f} "
            f"{result['counts']['accepted_proposals']:>10d}"
        )
    print(f"\n[E5-v2] Results written to: {output_path}")


if __name__ == "__main__":
    main()
