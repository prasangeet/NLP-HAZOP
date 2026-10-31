from __future__ import annotations

import pytest

from app.api.routes.experiments import aggregate_suite_metrics
from scripts.run_benchmark import (
    build_matched_case_summary,
    merge_candidate_predictions,
)


def _run(seed: int, f1: float, *, hops: int = 2) -> dict[str, object]:
    return {
        "dataset": "t301-reconstructed",
        "seed": seed,
        "max_hops": hops,
        "method_summary": {
            "graph_reasoning": {
                "micro_f1": f1,
                "coverage": 0.5,
                "status": "evaluated",
            },
            "matched_case_summary": {
                "graph_reasoning": {
                    "micro_f1": f1 / 2,
                    "coverage": 0.25,
                    "status": "evaluated",
                },
            },
            "prompt_only_llm": {
                "status": "skipped_missing_groq_api_key",
                "n_evaluated": 0,
            },
        },
    }


def test_aggregate_suite_metrics_reports_mean_and_student_t_interval() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2), _run(2, 0.4)])
    metric = result["full_test|t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]

    assert metric["n_seeds"] == 2
    assert metric["mean"] == pytest.approx(0.3)
    assert metric["sample_sd"] > 0
    assert metric["ci95_low"] < metric["mean"] < metric["ci95_high"]
    assert metric["interval_method"] == "Student-t 95% CI"


def test_aggregate_suite_metrics_marks_single_seed_interval_not_estimable() -> None:
    result = aggregate_suite_metrics([_run(42, 0.25)])
    metric = result["full_test|t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]

    assert metric["n_seeds"] == 1
    assert metric["mean"] == 0.25
    assert metric["sample_sd"] is None
    assert metric["ci95_low"] is None
    assert metric["ci95_high"] is None
    assert metric["interval_method"] == "not estimable with one seed"


def test_aggregate_suite_metrics_keeps_hop_settings_separate() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2, hops=2), _run(1, 0.8, hops=4)])

    assert result["full_test|t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]["mean"] == 0.2
    assert result["full_test|t301-reconstructed|hops=4|graph_reasoning"]["metrics"]["micro_f1"]["mean"] == 0.8


def test_aggregate_suite_metrics_does_not_turn_skipped_methods_into_zero_scores() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2)])

    assert "t301-reconstructed|hops=2|prompt_only_llm" not in result


def test_graph_tfidf_union_deduplicates_without_relabelling_provenance() -> None:
    graph = [
        {"value": "High pressure", "origin": "graph_inferred", "evidence": "path"},
    ]
    retrieval = [
        {"value": " high   pressure ", "origin": "retrieval_inferred", "evidence": "tfidf"},
        {"value": "Valve failure", "origin": "retrieval_inferred", "evidence": "similar row"},
    ]

    merged = merge_candidate_predictions(graph, retrieval)

    assert [item["value"] for item in merged] == [
        "High pressure",
        "Valve failure",
    ]
    assert merged[0]["origin"] == "graph_inferred"
    assert merged[1]["origin"] == "retrieval_inferred"



def test_matched_case_summary_scores_every_method_on_identical_case_ids() -> None:
    method_cases = {
        "tfidf": [
            {"case_id": "case-1", "gold_values": ["A"], "predictions": [{"value": "A"}], "latency_ms": 1.0},
            {"case_id": "case-2", "gold_values": ["B"], "predictions": [{"value": "X"}], "latency_ms": 1.0},
            {"case_id": "case-3", "gold_values": ["C"], "predictions": [{"value": "C"}], "latency_ms": 1.0},
        ],
        "llm": [
            {"case_id": "case-1", "gold_values": ["A"], "predictions": [{"value": "A"}], "latency_ms": 3.0},
            {"case_id": "case-2", "gold_values": ["B"], "predictions": [], "latency_ms": 3.0},
        ],
    }

    summary = build_matched_case_summary(method_cases, {"case-1", "case-2"})

    assert summary["tfidf"]["n_evaluated"] == 2
    assert summary["llm"]["n_evaluated"] == 2
    assert summary["tfidf"]["n_total_comparison_cases"] == 2
    assert summary["llm"]["n_total_comparison_cases"] == 2
    assert summary["tfidf"]["micro_f1"] == pytest.approx(0.5)
    assert summary["llm"]["micro_f1"] == pytest.approx(0.6666667)
