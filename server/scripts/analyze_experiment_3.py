
"""Case-level error analysis for E3 edge-support consensus.

Reads the existing E3 results. Does not rerun retrieval or recovery.

Outputs:
    results/e3/analysis/e3_case_analysis.json
    results/e3/analysis/e3_case_comparison.csv
    results/e3/analysis/e3_analysis_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT / "results" / "e3" / "edge_support_consensus.json"
)

DEFAULT_OUTPUT_DIR = ROOT / "results" / "e3" / "analysis"

BASELINE = "edge_support_1_1"

COMPARISONS = (
    "cause_consensus_2_1",
    "consequence_consensus_1_2",
    "both_edges_consensus_2_2",
)

# Prediction objects from different recovery implementations may use
# different field names. Strings are handled directly.
PREDICTION_FIELDS = (
    "value",
    "candidate",
    "entity",
    "endpoint",
    "object",
    "label",
    "text",
    "name",
)


def normalize(value: Any) -> str:
    """Normalize a label for case-insensitive comparison."""
    if value is None:
        return ""

    value = getattr(value, "value", value)

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def extract_prediction_value(prediction: Any) -> str:
    """Extract a comparable entity label from a prediction."""

    if isinstance(prediction, str):
        return prediction.strip()

    if isinstance(prediction, (int, float)):
        return str(prediction)

    if isinstance(prediction, dict):
        for field in PREDICTION_FIELDS:
            if field in prediction and prediction[field] is not None:
                return extract_prediction_value(prediction[field])

        raise ValueError(
            "Could not identify prediction value in dictionary: "
            f"{prediction!r}"
        )

    if prediction is None:
        return ""

    for field in PREDICTION_FIELDS:
        if hasattr(prediction, field):
            value = getattr(prediction, field)
            if value is not None:
                return extract_prediction_value(value)

    return str(prediction).strip()


def prediction_map(predictions: Any) -> dict[str, str]:
    """Return normalized prediction -> original display value."""

    if predictions is None:
        return {}

    if isinstance(predictions, (str, dict)):
        predictions = [predictions]

    if not isinstance(predictions, (list, tuple, set)):
        raise TypeError(
            "Expected predictions to be a list, tuple, set, string, "
            f"or dictionary; got {type(predictions).__name__}"
        )

    result = {}

    for prediction in predictions:
        display_value = extract_prediction_value(prediction)
        key = normalize(display_value)

        if key:
            result.setdefault(key, display_value)

    return result


def value_set(values: Any) -> dict[str, str]:
    """Normalize a list of gold values while preserving display labels."""

    if values is None:
        return {}

    if isinstance(values, str):
        values = [values]

    result = {}

    for value in values:
        display_value = str(value).strip()
        key = normalize(display_value)

        if key:
            result.setdefault(key, display_value)

    return result


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_micro_metrics(cases: list[dict]) -> dict:
    """Recalculate metrics from saved case predictions."""

    tp = 0
    fp = 0
    fn = 0
    evaluated = 0

    for case in cases:
        gold = value_set(case.get("gold_values", []))
        predicted = prediction_map(case.get("predictions", []))

        if predicted:
            evaluated += 1

        gold_keys = set(gold)
        predicted_keys = set(predicted)

        tp += len(gold_keys & predicted_keys)
        fp += len(predicted_keys - gold_keys)
        fn += len(gold_keys - predicted_keys)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    total = len(cases)
    coverage = evaluated / total if total else 0.0

    return {
        "cases": total,
        "evaluated_cases": evaluated,
        "abstained_cases": total - evaluated,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage if total else 0.0,
    }


def index_conditions(data: dict) -> dict[str, dict[str, dict]]:
    """Index condition cases by case ID and validate their identity."""

    conditions = data.get("conditions")

    if not isinstance(conditions, dict):
        raise ValueError("Input has no valid 'conditions' object.")

    required = (BASELINE, *COMPARISONS)

    missing = [name for name in required if name not in conditions]

    if missing:
        raise ValueError(
            "Missing required E3 conditions: " + ", ".join(missing)
        )

    indexed = {}

    for condition_name in required:
        condition = conditions[condition_name]
        cases = condition.get("cases")

        if not isinstance(cases, list):
            raise ValueError(
                f"{condition_name} has no valid cases list."
            )

        case_index = {}

        for case in cases:
            case_id = case.get("case_id")

            if not case_id:
                raise ValueError(
                    f"{condition_name} contains a case without case_id."
                )

            if case_id in case_index:
                raise ValueError(
                    f"Duplicate case_id {case_id!r} in {condition_name}."
                )

            case_index[case_id] = case

        indexed[condition_name] = case_index

    baseline_ids = set(indexed[BASELINE])

    for condition_name in COMPARISONS:
        condition_ids = set(indexed[condition_name])

        if condition_ids != baseline_ids:
            missing_ids = sorted(baseline_ids - condition_ids)
            extra_ids = sorted(condition_ids - baseline_ids)

            raise ValueError(
                f"Case mismatch in {condition_name}. "
                f"Missing={missing_ids}; extra={extra_ids}"
            )

    return indexed


def analyze_case(
    baseline_case: dict,
    comparison_case: dict,
    comparison_name: str,
) -> dict:
    """Compare one condition against the (1,1) baseline."""

    baseline_gold = value_set(
        baseline_case.get("gold_values", [])
    )
    comparison_gold = value_set(
        comparison_case.get("gold_values", [])
    )

    if set(baseline_gold) != set(comparison_gold):
        raise ValueError(
            "Gold labels differ across conditions for "
            f"{baseline_case['case_id']}."
        )

    baseline_predictions = prediction_map(
        baseline_case.get("predictions", [])
    )
    comparison_predictions = prediction_map(
        comparison_case.get("predictions", [])
    )

    gold_keys = set(baseline_gold)
    baseline_keys = set(baseline_predictions)
    comparison_keys = set(comparison_predictions)

    baseline_tp = gold_keys & baseline_keys
    comparison_tp = gold_keys & comparison_keys

    baseline_fp = baseline_keys - gold_keys
    comparison_fp = comparison_keys - gold_keys

    retained_gold = baseline_tp & comparison_tp
    lost_gold = baseline_tp - comparison_tp
    recovered_gold = comparison_tp - baseline_tp

    retained_predictions = baseline_keys & comparison_keys
    removed_predictions = baseline_keys - comparison_keys
    added_predictions = comparison_keys - baseline_keys

    baseline_audit = baseline_case.get("global_edge_audit", {})
    comparison_audit = comparison_case.get("global_edge_audit", {})

    baseline_cause_qualifying = baseline_audit.get(
        "qualifying_cause_edges"
    )
    comparison_cause_qualifying = comparison_audit.get(
        "qualifying_cause_edges"
    )

    baseline_consequence_qualifying = baseline_audit.get(
        "qualifying_consequence_edges"
    )
    comparison_consequence_qualifying = comparison_audit.get(
        "qualifying_consequence_edges"
    )

    return {
        "case_id": baseline_case["case_id"],
        "scenario_id": baseline_case.get("scenario_id"),
        "known_endpoint": baseline_case.get("known_endpoint"),
        "comparison": comparison_name,
        "gold_count": len(gold_keys),
        "baseline_prediction_count": len(baseline_keys),
        "comparison_prediction_count": len(comparison_keys),
        "baseline_true_positives": len(baseline_tp),
        "comparison_true_positives": len(comparison_tp),
        "baseline_false_positives": len(baseline_fp),
        "comparison_false_positives": len(comparison_fp),
        "retained_gold": [
            baseline_gold[key] for key in sorted(retained_gold)
        ],
        "lost_gold": [
            baseline_gold[key] for key in sorted(lost_gold)
        ],
        "newly_recovered_gold": [
            comparison_gold[key] for key in sorted(recovered_gold)
        ],
        "baseline_false_positives": [
            baseline_predictions[key] for key in sorted(baseline_fp)
        ],
        "comparison_false_positives": [
            comparison_predictions[key] for key in sorted(comparison_fp)
        ],
        "removed_predictions": [
            baseline_predictions[key]
            for key in sorted(removed_predictions)
        ],
        "added_predictions": [
            comparison_predictions[key]
            for key in sorted(added_predictions)
        ],
        "retained_predictions": [
            baseline_predictions[key]
            for key in sorted(retained_predictions)
        ],
        "baseline_cause_edges_qualifying": baseline_cause_qualifying,
        "comparison_cause_edges_qualifying": comparison_cause_qualifying,
        "baseline_consequence_edges_qualifying": (
            baseline_consequence_qualifying
        ),
        "comparison_consequence_edges_qualifying": (
            comparison_consequence_qualifying
        ),
        "baseline_qualifying_paths": baseline_audit.get(
            "qualifying_complete_paths"
        ),
        "comparison_qualifying_paths": comparison_audit.get(
            "qualifying_complete_paths"
        ),
        "baseline_retrieved_edges": baseline_audit.get(
            "raw_retrieved_edges"
        ),
        "comparison_retrieved_edges": comparison_audit.get(
            "raw_retrieved_edges"
        ),
        "baseline_retained_edges": baseline_audit.get(
            "retained_retrieved_edges"
        ),
        "comparison_retained_edges": comparison_audit.get(
            "retained_retrieved_edges"
        ),
        "baseline_abstained": not bool(baseline_keys),
        "comparison_abstained": not bool(comparison_keys),
        "cause_edge_count_decreased": (
            baseline_cause_qualifying is not None
            and comparison_cause_qualifying is not None
            and comparison_cause_qualifying
            < baseline_cause_qualifying
        ),
        "consequence_edge_count_decreased": (
            baseline_consequence_qualifying is not None
            and comparison_consequence_qualifying is not None
            and comparison_consequence_qualifying
            < baseline_consequence_qualifying
        ),
        "interpretation_limit": (
            "Aggregate edge counts cannot identify the exact support "
            "of an individual lost gold relationship."
        ),
    }


def summarize_comparisons(
    analyses: dict[str, list[dict]],
) -> dict:
    """Aggregate case-level changes for each comparison."""

    summaries = {}

    for name, rows in analyses.items():
        lost = sum(len(row["lost_gold"]) for row in rows)
        recovered = sum(
            len(row["newly_recovered_gold"]) for row in rows
        )
        retained = sum(len(row["retained_gold"]) for row in rows)

        removed = sum(
            len(row["removed_predictions"]) for row in rows
        )
        added = sum(
            len(row["added_predictions"]) for row in rows
        )

        summaries[name] = {
            "cases": len(rows),
            "cases_with_lost_gold": sum(
                bool(row["lost_gold"]) for row in rows
            ),
            "cases_with_newly_recovered_gold": sum(
                bool(row["newly_recovered_gold"]) for row in rows
            ),
            "cases_with_any_prediction_change": sum(
                bool(row["removed_predictions"] or row["added_predictions"])
                for row in rows
            ),
            "cases_that_became_abstentions": sum(
                not row["baseline_abstained"]
                and row["comparison_abstained"]
                for row in rows
            ),
            "cases_that_stopped_abstaining": sum(
                row["baseline_abstained"]
                and not row["comparison_abstained"]
                for row in rows
            ),
            "gold_retained": retained,
            "gold_lost": lost,
            "gold_newly_recovered": recovered,
            "predictions_removed": removed,
            "predictions_added": added,
            "cases_with_fewer_qualifying_cause_edges": sum(
                row["cause_edge_count_decreased"] for row in rows
            ),
            "cases_with_fewer_qualifying_consequence_edges": sum(
                row["consequence_edge_count_decreased"] for row in rows
            ),
        }

    return summaries


def verify_saved_metrics(
    indexed: dict[str, dict[str, dict]],
    conditions: dict,
) -> dict:
    """Compare recomputed metrics with the metrics stored by E3."""

    checks = {}

    for name, case_index in indexed.items():
        cases = list(case_index.values())
        recomputed = calculate_micro_metrics(cases)

        saved = conditions[name].get("summary", {})

        fields = (
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "coverage",
            "abstention_rate",
        )

        comparisons = {}

        for field in fields:
            saved_value = safe_float(saved.get(field))
            calculated_value = recomputed[field]

            comparisons[field] = {
                "saved": saved_value,
                "recomputed": calculated_value,
                "matches": (
                    saved_value is not None
                    and abs(saved_value - calculated_value) < 1e-4
                ),
            }

        checks[name] = {
            "recomputed_metrics": recomputed,
            "saved_metrics": saved,
            "checks": comparisons,
            "all_match": all(
                item["matches"] for item in comparisons.values()
            ),
        }

    return checks


def write_case_csv(
    path: Path,
    analyses: dict[str, list[dict]],
) -> None:
    """Write one row per case per comparison."""

    fields = [
        "case_id",
        "scenario_id",
        "known_endpoint",
        "comparison",
        "gold_count",
        "baseline_prediction_count",
        "comparison_prediction_count",
        "baseline_true_positives",
        "comparison_true_positives",
        "baseline_false_positives",
        "comparison_false_positives",
        "retained_gold",
        "lost_gold",
        "newly_recovered_gold",
        "baseline_false_positives_values",
        "comparison_false_positives_values",
        "removed_predictions",
        "added_predictions",
        "retained_predictions",
        "baseline_cause_edges_qualifying",
        "comparison_cause_edges_qualifying",
        "baseline_consequence_edges_qualifying",
        "comparison_consequence_edges_qualifying",
        "baseline_qualifying_paths",
        "comparison_qualifying_paths",
        "baseline_retrieved_edges",
        "comparison_retrieved_edges",
        "baseline_retained_edges",
        "comparison_retained_edges",
        "baseline_abstained",
        "comparison_abstained",
    ]

    list_fields = {
        "retained_gold",
        "lost_gold",
        "newly_recovered_gold",
        "baseline_false_positives_values",
        "comparison_false_positives_values",
        "removed_predictions",
        "added_predictions",
        "retained_predictions",
    }

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for rows in analyses.values():
            for row in rows:
                output = dict(row)

                output["baseline_false_positives_values"] = (
                    output.pop("baseline_false_positives")
                )
                output["comparison_false_positives_values"] = (
                    output.pop("comparison_false_positives")
                )

                for field in list_fields:
                    output[field] = json.dumps(
                        output.get(field, []),
                        ensure_ascii=False,
                    )

                writer.writerow({
                    field: output.get(field)
                    for field in fields
                })


def write_report(
    path: Path,
    summaries: dict,
    metric_checks: dict,
) -> None:
    lines = [
        "# E3 Case-Level Error Analysis",
        "",
        "Baseline: `edge_support_1_1`.",
        "",
        "## Comparison summary",
        "",
        "| Condition | Cases losing gold | Gold lost | Gold newly recovered "
        "| Predictions removed | Predictions added | Became abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` "
            f"| {summary['cases_with_lost_gold']} "
            f"| {summary['gold_lost']} "
            f"| {summary['gold_newly_recovered']} "
            f"| {summary['predictions_removed']} "
            f"| {summary['predictions_added']} "
            f"| {summary['cases_that_became_abstentions']} |"
        )

    lines.extend([
        "",
        "## Saved metric consistency",
        "",
        "| Condition | Precision | Recall | F1 | Coverage | Abstention |",
        "|---|---|---|---|---|---|",
    ])

    for name, check in metric_checks.items():
        values = check["checks"]

        def status(field: str) -> str:
            item = values[field]
            if item["saved"] is None:
                return "missing"
            return "match" if item["matches"] else "MISMATCH"

        lines.append(
            f"| `{name}` "
            f"| {status('micro_precision')} "
            f"| {status('micro_recall')} "
            f"| {status('micro_f1')} "
            f"| {status('coverage')} "
            f"| {status('abstention_rate')} |"
        )

    lines.extend([
        "",
        "## Interpretation cautions",
        "",
        "- A lost gold value means it was predicted in the (1,1) "
        "condition but not in the comparison condition.",
        "- A removed prediction is not automatically a false positive; "
        "it may be a correct gold prediction or an incorrect prediction.",
        "- Aggregate qualifying-edge counts do not reveal the support "
        "count of the exact edge associated with an individual gold value.",
        "- This analysis is based on reconstructed row-level silver labels, "
        "not expert-validated causal ground truth.",
        "- Scenario recurrence does not establish independent evidence "
        "or physical causality.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze E3 case-level prediction changes."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the existing E3 JSON result.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for analysis outputs.",
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"E3 result file not found: {input_path}"
        )

    print(f"[E3 Analysis] Reading: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    if data.get("metadata", {}).get("experiment") != "E3":
        raise ValueError(
            "Input metadata does not identify this as experiment E3."
        )

    indexed = index_conditions(data)
    conditions = data["conditions"]

    baseline_cases = indexed[BASELINE]

    analyses = {}

    for comparison_name in COMPARISONS:
        rows = []

        for case_id in sorted(baseline_cases):
            baseline_case = baseline_cases[case_id]
            comparison_case = indexed[comparison_name][case_id]

            rows.append(
                analyze_case(
                    baseline_case,
                    comparison_case,
                    comparison_name,
                )
            )

        analyses[comparison_name] = rows

    summaries = summarize_comparisons(analyses)

    metric_checks = verify_saved_metrics(
        indexed,
        conditions,
    )

    output = {
        "metadata": {
            "analysis": "E3 case-level error analysis",
            "source_file": str(input_path),
            "baseline": BASELINE,
            "comparisons": list(COMPARISONS),
            "case_count": len(baseline_cases),
            "comparison_count": len(COMPARISONS),
            "normalization": (
                "Case-insensitive; leading, trailing, and repeated "
                "whitespace normalized."
            ),
        },
        "baseline_metrics": calculate_micro_metrics(
            list(baseline_cases.values())
        ),
        "metric_consistency_checks": metric_checks,
        "comparison_summaries": summaries,
        "case_analysis": analyses,
        "limitations": [
            (
                "The saved E3 audit contains aggregate qualifying edge "
                "counts, not support counts for each individual edge."
            ),
            (
                "Consequently, this analysis cannot conclusively attribute "
                "a lost gold consequence to the support of its exact "
                "cause-to-deviation or deviation-to-consequence edge."
            ),
            (
                "Predictions disappearing after filtering do not by "
                "themselves establish whether those predictions were "
                "correct or incorrect."
            ),
            (
                "The reconstructed dataset provides silver labels, "
                "not expert-validated causal ground truth."
            ),
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "e3_case_analysis.json"
    csv_path = output_dir / "e3_case_comparison.csv"
    report_path = output_dir / "e3_analysis_report.md"

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_case_csv(csv_path, analyses)
    write_report(report_path, summaries, metric_checks)

    print("\n========== E3 CASE-LEVEL ANALYSIS ==========")

    for name, summary in summaries.items():
        print(f"\n{name}")
        print(
            f"  Cases losing gold: "
            f"{summary['cases_with_lost_gold']}"
        )
        print(f"  Gold values lost: {summary['gold_lost']}")
        print(
            f"  Gold values newly recovered: "
            f"{summary['gold_newly_recovered']}"
        )
        print(
            f"  Predictions removed: "
            f"{summary['predictions_removed']}"
        )
        print(
            f"  Predictions added: "
            f"{summary['predictions_added']}"
        )
        print(
            f"  Cases becoming abstentions: "
            f"{summary['cases_that_became_abstentions']}"
        )

    mismatches = [
        name
        for name, check in metric_checks.items()
        if not check["all_match"]
    ]

    print("\n========== METRIC CONSISTENCY ==========")

    if mismatches:
        print(
            "[WARNING] Saved metrics differ from recomputed metrics for: "
            + ", ".join(mismatches)
        )
    else:
        print("[OK] All saved aggregate metrics match recomputed metrics.")

    print("\n[E3 Analysis] Outputs:")
    print(f"  JSON:   {json_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
