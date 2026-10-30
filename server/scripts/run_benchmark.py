"""Run a leakage-aware, provenance-labelled HAZOP link-recovery benchmark.

The benchmark targets row-level cause/consequence association recovery.
Those labels are SILVER labels derived from fields co-listed in one HAZOP
worksheet row; they are not expert-validated causal ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.application.strategies.graph_reasoner import GraphReasoner
from app.application.strategies.hybrid_reasoner import HybridReasoner
from app.application.strategies.llm_reasoner import LLMReasoner
from app.application.strategies.prompt_only_reasoner import PromptOnlyReasoner
from app.application.strategies.retrieval.graph_neighborhood import (
    GraphNeighborhoodRetriever,
)
from app.application.strategies.retrieval.hybrid import HybridRetriever
from app.application.strategies.retrieval.scenario_local import (
    ScenarioLocalRetriever,
)
from app.application.strategies.retrieval.tfidf import TfidfRetriever
from app.application.use_cases.build_evidence_context import BuildEvidenceContext
from app.application.use_cases.recover_missing_link import (
    MissingEndpoint,
    RecoverMissingLink,
    RecoveryRequest,
)
from app.domain.entities.hazop_record import (
    HazopRecord,
    RecordType,
    RelationshipOrigin,
)
from app.domain.reasoning.models import ReasoningResult, ReasoningStatus
from app.infrastructure.datasets.hazop_csv_loader import HazopCsvLoader
from app.infrastructure.graph.in_memory_graph_repository import (
    InMemoryGraphRepository,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATHS = {
    "t301-reconstructed": ROOT / "data/raw/t301/T301_HAZOP_CCPS2001_style.csv",
    "t301-synthetic": ROOT / "data/raw/t301/T301_HAZOP_CCPS2001_style.csv",
    "dwsim-synthetic": ROOT / "data/raw/dwsim/synthetic_hazop_D001_D030.csv",
}
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One held-out endpoint query and its row-derived silver answer set."""

    case_id: str
    scenario_id: str
    known_endpoint: str
    gold_values: tuple[str, ...]
    record_type: str
    scenario_context: str


def normalize(value: str) -> str:
    return NORMALIZE_RE.sub(" ", value.casefold()).strip()


def stable_split(
    records: tuple[HazopRecord, ...],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[tuple[HazopRecord, ...], tuple[HazopRecord, ...]]:
    """Split by scenario ID, never by individual cause/consequence pairs."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = sorted(records, key=lambda record: record.scenario_id.casefold())
    if len(ordered) < 2:
        raise ValueError("At least two scenarios are required for a split")

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    test_count = min(len(shuffled) - 1, max(1, math.ceil(len(shuffled) * test_fraction)))
    test_ids = {record.scenario_id for record in shuffled[:test_count]}
    train = tuple(record for record in ordered if record.scenario_id not in test_ids)
    test = tuple(record for record in ordered if record.scenario_id in test_ids)
    return train, test


def make_cases(records: tuple[HazopRecord, ...]) -> tuple[BenchmarkCase, ...]:
    """Make one query per listed cause; all row consequences are silver targets."""

    cases: list[BenchmarkCase] = []
    for record in sorted(records, key=lambda item: item.scenario_id.casefold()):
        if not record.causes or not record.consequences:
            continue
        scenario_context = " | ".join(
            value
            for value in (
                record.plant_unit,
                record.equipment,
                record.design_intent,
                record.parameter,
                record.guideword,
                record.deviation,
            )
            if value
        )
        for index, cause in enumerate(record.causes, start=1):
            case_id = f"{record.scenario_id}::cause-{index:02d}"
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    scenario_id=record.scenario_id,
                    known_endpoint=cause,
                    gold_values=record.consequences,
                    record_type=record.record_type.value,
                    scenario_context=scenario_context,
                )
            )
    return tuple(cases)


def strip_target_labels(record: HazopRecord) -> HazopRecord:
    """Retain target metadata but remove all target answer/evidence fields."""

    return replace(
        record,
        causes=(),
        consequences=(),
        safeguards=(),
        recommendations=(),
        causal_relations=(),
        remarks="",
    )


def load_dataset(
    dataset: str,
    dataset_path: Path | None,
) -> tuple[HazopRecord, ...]:
    path = dataset_path or DATASET_PATHS[dataset]
    records = HazopCsvLoader().load(path, dataset_id=dataset)

    if dataset == "t301-reconstructed":
        records = tuple(
            record
            for record in records
            if record.record_type == RecordType.SOURCE_RECONSTRUCTED
        )
    elif dataset == "t301-synthetic":
        records = tuple(
            record
            for record in records
            if record.record_type == RecordType.SYNTHETIC
        )
    # DWSIM is a separate synthetic dataset and is never pooled with T-301.
    if not records:
        raise ValueError(f"No rows matched dataset selection: {dataset}")
    return records


def build_repository(
    train: tuple[HazopRecord, ...],
    test: tuple[HazopRecord, ...],
) -> InMemoryGraphRepository:
    repository = InMemoryGraphRepository()
    for record in train:
        repository.save_record(record)
    for record in test:
        # The held-out scenario remains addressable but contains no target
        # labels, graph edges, safeguards, recommendations, or remarks.
        repository.save_context_record(strip_target_labels(record))
    return repository


def prediction_from_relation(
    relation,
    missing_endpoint: str,
    *,
    scenario_id: str,
    scope: str = "unknown",
    relevance_score: float | None = None,
) -> dict[str, Any]:
    value = relation.subject if missing_endpoint == "subject" else relation.object
    return {
        "value": value,
        "origin": relation.origin.value,
        "source_id": relation.source_id,
        "evidence": relation.evidence,
        "scope": scope,
        "relevance_score": relevance_score,
        "scenario_id": scenario_id,
    }


def predictions_from_recovery(result, missing_endpoint: str) -> list[dict[str, Any]]:
    return [
        prediction_from_relation(
            candidate.relation,
            missing_endpoint,
            scenario_id=candidate.scenario_id,
            scope=candidate.scope.value,
            relevance_score=candidate.relevance_score,
        )
        for candidate in result.candidates
    ]


def prediction_from_reasoning(
    result: ReasoningResult,
    missing_endpoint: str,
    scenario_id: str,
) -> list[dict[str, Any]]:
    if (
        result.status != ReasoningStatus.PROPOSED
        or result.proposed_relation is None
    ):
        return []
    return [
        prediction_from_relation(
            result.proposed_relation,
            missing_endpoint,
            scenario_id=scenario_id,
        )
    ]


def score_predictions(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute set-recovery, ranking, coverage, evidence and latency metrics."""

    if not cases:
        return {
            "n_evaluated": 0,
            "error": "No cases evaluated",
        }

    total_tp = total_fp = total_fn = 0
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    reciprocal_ranks: list[float] = []
    top1_hits = 0
    covered = 0
    prediction_count = 0
    evidence_text_count = 0
    non_model_origin_count = 0
    source_or_expert_count = 0
    latency_values: list[float] = []
    origin_counts: Counter[str] = Counter()

    for case in cases:
        gold = {normalize(item) for item in case["gold_values"] if normalize(item)}
        predictions = case["predictions"]
        ranked_values = [
            normalize(item["value"])
            for item in predictions
            if normalize(item["value"])
        ]
        unique_ranked: list[str] = []
        for value in ranked_values:
            if value not in unique_ranked:
                unique_ranked.append(value)
        predicted = set(unique_ranked)

        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        rank = next(
            (index for index, value in enumerate(unique_ranked, start=1)
             if value in gold),
            None,
        )
        if rank == 1:
            top1_hits += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        if predictions:
            covered += 1
        prediction_count += len(predictions)

        for prediction in predictions:
            origin = prediction.get("origin", "unknown")
            origin_counts[origin] += 1
            if prediction.get("evidence", "").strip():
                evidence_text_count += 1
            if origin not in {
                RelationshipOrigin.MODEL_INFERRED.value,
            }:
                non_model_origin_count += 1
            if origin in {
                RelationshipOrigin.SOURCE.value,
                RelationshipOrigin.EXPERT.value,
            }:
                source_or_expert_count += 1

        latency_values.append(float(case.get("latency_ms", 0.0)))

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    denominator = prediction_count or 1
    n = len(cases)
    return {
        "n_evaluated": n,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": sum(precisions) / n,
        "macro_recall": sum(recalls) / n,
        "macro_f1": sum(f1s) / n,
        "hit_at_1": top1_hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "coverage": covered / n,
        "abstention_rate": 1.0 - covered / n,
        "mean_candidates": prediction_count / n,
        "evidence_text_rate": evidence_text_count / denominator,
        "non_model_origin_rate": non_model_origin_count / denominator,
        "source_or_expert_origin_rate": source_or_expert_count / denominator,
        "mean_latency_ms": (
            sum(latency_values) / len(latency_values)
            if latency_values
            else 0.0
        ),
        "origin_counts": dict(origin_counts),
        "cost_usd": None,
        "cost_note": "Not captured by the current provider wrapper.",
    }


def _run_recovery(
    use_case: RecoverMissingLink,
    builder: BuildEvidenceContext,
    request: RecoveryRequest,
    *,
    reasoner=None,
) -> tuple[list[dict[str, Any]], float, str]:
    started = time.perf_counter()
    recovery = use_case.execute(request)
    if reasoner is None:
        predictions = predictions_from_recovery(
            recovery,
            request.missing_endpoint.value,
        )
        status = recovery.status.value
    else:
        context = builder.execute(recovery, request)
        reasoning = reasoner.reason(context)
        predictions = prediction_from_reasoning(
            reasoning,
            request.missing_endpoint.value,
            request.scenario_id,
        )
        status = (
            reasoning.abstention_reason.value
            if reasoning.status == ReasoningStatus.ABSTAINED
            and reasoning.abstention_reason is not None
            else reasoning.status.value
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return predictions, latency_ms, status


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected_path = Path(args.dataset_path).resolve() if args.dataset_path else None
    records = load_dataset(args.dataset, selected_path)
    train, test = stable_split(
        records,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    cases = list(make_cases(test))
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError(
            "The held-out split produced no cause/consequence queries. "
            "Increase the dataset size or adjust --test-fraction."
        )

    repository = build_repository(train, test)
    builder = BuildEvidenceContext()
    local_retriever = ScenarioLocalRetriever(repository)
    graph_retriever = GraphNeighborhoodRetriever(
        local_retriever,
        repository,
    )
    vector_retriever = TfidfRetriever(repository)
    hybrid_retriever = HybridRetriever(
        graph_retriever,
        vector_retriever,
    )

    local_use_case = RecoverMissingLink(local_retriever)
    graph_use_case = RecoverMissingLink(graph_retriever)
    vector_use_case = RecoverMissingLink(vector_retriever)
    hybrid_use_case = RecoverMissingLink(hybrid_retriever)

    graph_reasoner = GraphReasoner(max_hops=args.max_hops)

    llm_available = bool(os.getenv("GROQ_API_KEY"))
    llm_enabled = args.with_llm and llm_available
    llm_reasoner = LLMReasoner(model_name=args.model) if llm_enabled else None
    prompt_reasoner = PromptOnlyReasoner(model_name=args.model) if llm_enabled else None
    hybrid_reasoner = (
        HybridReasoner(graph_reasoner, llm_reasoner)
        if llm_enabled and llm_reasoner is not None
        else None
    )

    llm_case_ids = {
        case.case_id
        for case in cases[: args.max_llm_cases]
    }

    per_case: list[dict[str, Any]] = []
    method_cases: dict[str, list[dict[str, Any]]] = {
        "prompt_only_llm": [],
        "tfidf_vector_retrieval": [],
        "graph_reasoning": [],
        "hybrid_graph_vector_llm": [],
    }

    for case in cases:
        request = RecoveryRequest(
            scenario_id=case.scenario_id,
            predicate="causes",
            known_endpoint=case.known_endpoint,
            missing_endpoint=MissingEndpoint.OBJECT,
            limit=args.top_k,
        )
        case_result: dict[str, Any] = {
            "case_id": case.case_id,
            "scenario_id": case.scenario_id,
            "record_type": case.record_type,
            "known_endpoint": case.known_endpoint,
            "gold_values": list(case.gold_values),
            "silver_label_definition": (
                "Consequences co-listed in the same held-out worksheet row; "
                "not independently expert-validated."
            ),
            "methods": {},
        }

        # Vector-only retrieval baseline.
        vector_predictions, vector_latency, vector_status = _run_recovery(
            vector_use_case,
            builder,
            request,
        )
        vector_case = {
            "case_id": case.case_id,
            "gold_values": list(case.gold_values),
            "predictions": vector_predictions,
            "latency_ms": vector_latency,
            "status": vector_status,
        }
        method_cases["tfidf_vector_retrieval"].append(vector_case)
        case_result["methods"]["tfidf_vector_retrieval"] = vector_case

        # Graph baseline: training-scenario neighbourhood only. The target
        # record has no edges in the repository, preventing same-row leakage.
        graph_predictions, graph_latency, graph_status = _run_recovery(
            graph_use_case,
            builder,
            request,
            reasoner=graph_reasoner,
        )
        graph_case = {
            "case_id": case.case_id,
            "gold_values": list(case.gold_values),
            "predictions": graph_predictions,
            "latency_ms": graph_latency,
            "status": graph_status,
        }
        method_cases["graph_reasoning"].append(graph_case)
        case_result["methods"]["graph_reasoning"] = graph_case

        # LLM methods are evaluated on the same deterministic capped subset.
        if llm_enabled and case.case_id in llm_case_ids:
            prompt_predictions, prompt_latency, prompt_status = _run_recovery(
                local_use_case,
                builder,
                request,
                reasoner=prompt_reasoner,
            )
            prompt_case = {
                "case_id": case.case_id,
                "gold_values": list(case.gold_values),
                "predictions": prompt_predictions,
                "latency_ms": prompt_latency,
                "status": prompt_status,
            }
            method_cases["prompt_only_llm"].append(prompt_case)
            case_result["methods"]["prompt_only_llm"] = prompt_case

            hybrid_predictions, hybrid_latency, hybrid_status = _run_recovery(
                hybrid_use_case,
                builder,
                request,
                reasoner=hybrid_reasoner,
            )
            hybrid_case = {
                "case_id": case.case_id,
                "gold_values": list(case.gold_values),
                "predictions": hybrid_predictions,
                "latency_ms": hybrid_latency,
                "status": hybrid_status,
            }
            method_cases["hybrid_graph_vector_llm"].append(hybrid_case)
            case_result["methods"]["hybrid_graph_vector_llm"] = hybrid_case
        else:
            skipped_reason = (
                "llm_call_cap"
                if llm_enabled
                else (
                    "missing_groq_api_key"
                    if args.with_llm and not llm_available
                    else "disabled"
                )
            )
            for method in ("prompt_only_llm", "hybrid_graph_vector_llm"):
                case_result["methods"][method] = {
                    "status": f"not_evaluated_{skipped_reason}",
                    "predictions": [],
                    "latency_ms": 0.0,
                }

        per_case.append(case_result)

    method_summary: dict[str, Any] = {}
    for method, evaluated_cases in method_cases.items():
        if evaluated_cases:
            method_summary[method] = {
                "status": "evaluated",
                **score_predictions(evaluated_cases),
                "n_total_test_cases": len(cases),
            }
        elif method in {"prompt_only_llm", "hybrid_graph_vector_llm"}:
            method_summary[method] = {
                "status": (
                    "skipped_missing_groq_api_key"
                    if args.with_llm and not llm_available
                    else "disabled_or_call_cap_zero"
                ),
                "n_evaluated": 0,
                "n_total_test_cases": len(cases),
            }
        else:
            method_summary[method] = {
                "status": "no_predictions_evaluated",
                "n_evaluated": 0,
                "n_total_test_cases": len(cases),
            }

    origin_counts = Counter(record.record_type.value for record in records)
    validation_notes = Counter()
    for record in records:
        match = re.search(r"Validation note:\s*(.+)$", record.remarks)
        if match:
            validation_notes[match.group(1).strip()] += 1

    metadata = {
        "project": "NLP-HAZOP",
        "dataset": args.dataset,
        "dataset_path": str(selected_path or DATASET_PATHS[args.dataset]),
        "dataset_rows": len(records),
        "train_scenarios": len(train),
        "test_scenarios": len(test),
        "test_queries": len(cases),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "top_k": args.top_k,
        "max_graph_hops": args.max_hops,
        "tfidf_index_build_ms": vector_retriever.index_build_ms,
        "method": "scenario-level holdout; target labels and target graph edges removed from index",
        "label_status": "silver_row_association",
        "label_warning": (
            "Labels are cause/consequence values co-listed in the same source "
            "or synthetic worksheet row. They are not independent causal "
            "ground truth and none of the supplied rows is expert-validated."
        ),
        "record_type_counts": dict(origin_counts),
        "validation_note_counts": dict(validation_notes),
        "llm_requested": args.with_llm,
        "llm_enabled": llm_enabled,
        "llm_model": args.model if llm_enabled else None,
        "llm_max_cases": args.max_llm_cases,
        "llm_calls_per_evaluated_case_max": 2,
        "cost_measurement": "not captured; cost_usd remains null",
        "created_at": datetime.now(UTC).isoformat(),
    }

    return {
        "metadata": metadata,
        "method_summary": method_summary,
        "cases": per_case,
    }


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "status",
            "n_evaluated",
            "n_total_test_cases",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "hit_at_1",
            "mrr",
            "coverage",
            "abstention_rate",
            "mean_candidates",
            "evidence_text_rate",
            "non_model_origin_rate",
            "source_or_expert_origin_rate",
            "mean_latency_ms",
            "cost_usd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in result["method_summary"].items():
            writer.writerow(
                {
                    "method": method,
                    **{
                        field: metrics.get(field)
                        for field in fieldnames
                        if field != "method"
                    },
                }
            )

    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "case_id",
            "scenario_id",
            "record_type",
            "known_endpoint",
            "gold_values",
            "method",
            "status",
            "predicted_values",
            "predicted_origins",
            "latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in result["cases"]:
            for method, value in case["methods"].items():
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "scenario_id": case["scenario_id"],
                        "record_type": case["record_type"],
                        "known_endpoint": case["known_endpoint"],
                        "gold_values": " | ".join(case["gold_values"]),
                        "method": method,
                        "status": value.get("status", ""),
                        "predicted_values": " | ".join(
                            item["value"] for item in value.get("predictions", [])
                        ),
                        "predicted_origins": " | ".join(
                            item.get("origin", "")
                            for item in value.get("predictions", [])
                        ),
                        "latency_ms": value.get("latency_ms", 0.0),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PATHS),
        required=True,
        help="Run one dataset track only; datasets are never pooled.",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--max-llm-cases", type=int, default=15)
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    args = parser.parse_args()
    if args.max_llm_cases < 0:
        parser.error("--max-llm-cases must be >= 0")
    if args.limit_cases is not None and args.limit_cases < 1:
        parser.error("--limit-cases must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_hops < 2:
        parser.error("--max-hops must be >= 2")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "results" / args.dataset
    )
    write_outputs(result, output_dir)
    print(json.dumps(result["method_summary"], indent=2))
    print(f"\nWrote benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
