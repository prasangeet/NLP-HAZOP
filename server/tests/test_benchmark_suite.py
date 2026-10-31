from __future__ import annotations

import pytest\n\nfrom app.api.routes.experiments import aggregate_suite_metrics


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
            "prompt_only_llm": {
                "status": "skipped_missing_groq_api_key",
                "n_evaluated": 0,
            },
        },
    }


def test_aggregate_suite_metrics_reports_mean_and_student_t_interval() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2), _run(2, 0.4)])
    metric = result["t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]

    assert metric["n_seeds"] == 2
    assert metric["mean"] == pytest.approx(0.3)
    assert metric["sample_sd"] > 0
    assert metric["ci95_low"] < metric["mean"] < metric["ci95_high"]
    assert metric["interval_method"] == "Student-t 95% CI"


def test_aggregate_suite_metrics_marks_single_seed_interval_not_estimable() -> None:
    result = aggregate_suite_metrics([_run(42, 0.25)])
    metric = result["t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]

    assert metric["n_seeds"] == 1
    assert metric["mean"] == 0.25
    assert metric["sample_sd"] is None
    assert metric["ci95_low"] is None
    assert metric["ci95_high"] is None
    assert metric["interval_method"] == "not estimable with one seed"


def test_aggregate_suite_metrics_keeps_hop_settings_separate() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2, hops=2), _run(1, 0.8, hops=4)])

    assert result["t301-reconstructed|hops=2|graph_reasoning"]["metrics"]["micro_f1"]["mean"] == 0.2
    assert result["t301-reconstructed|hops=4|graph_reasoning"]["metrics"]["micro_f1"]["mean"] == 0.8


def test_aggregate_suite_metrics_does_not_turn_skipped_methods_into_zero_scores() -> None:
    result = aggregate_suite_metrics([_run(1, 0.2)])

    assert "t301-reconstructed|hops=2|prompt_only_llm" not in result
