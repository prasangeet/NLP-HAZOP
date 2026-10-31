"""Research benchmark endpoints for the Experiments workbench."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import math
from collections import defaultdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/experiments", tags=["experiments"])
SERVER_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = SERVER_ROOT.parent
BENCHMARK_SCRIPT = SERVER_ROOT / "scripts" / "run_benchmark.py"
RUNS_ROOT = SERVER_ROOT / "data" / "experiments" / "runs"


class BenchmarkRequest(BaseModel):
    dataset: Literal["t301-reconstructed", "t301-synthetic", "dwsim-synthetic"] = "t301-reconstructed"
    seed: int = Field(default=42, ge=0, le=1000000)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    with_llm: bool = False
    max_llm_cases: int = Field(default=15, ge=0, le=100)


@router.post("/benchmark", summary="Run the reproducible HAZOP recovery benchmark")
def run_benchmark(payload: BenchmarkRequest) -> dict[str, object]:
    """Run the existing benchmark script and return its real artifacts."""
    if not BENCHMARK_SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="Benchmark script is not available in this installation.")

    with tempfile.TemporaryDirectory(prefix="nlp-hazop-benchmark-") as output:
        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--dataset",
            payload.dataset,
            "--seed",
            str(payload.seed),
            "--test-fraction",
            str(payload.test_fraction),
            "--output-dir",
            output,
        ]
        if payload.with_llm:
            command.extend(["--with-llm", "--max-llm-cases", str(payload.max_llm_cases)])

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SERVER_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            completed = subprocess.run(
                command,
                cwd=SERVER_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=504,
                detail="Benchmark exceeded the 300-second API limit. Run it from the CLI for longer experiments.",
            ) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not start benchmark: {exc}") from exc

        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Benchmark failed.").strip()
            raise HTTPException(status_code=500, detail=message[-6000:])

        results_path = Path(output) / "results.json"
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Benchmark completed but its results.json artifact could not be read.") from exc

        return {
            "status": "completed",
            "dataset": payload.dataset,
            "seed": payload.seed,
            "test_fraction": payload.test_fraction,
            "method_summary": results.get("method_summary", {}),
            "metadata": results.get("metadata", {}),
            "cases": results.get("cases", []),
            "artifacts": {
                "results_json": results,
                "summary_csv": (Path(output) / "summary.csv").read_text(encoding="utf-8"),
                "predictions_csv": (Path(output) / "predictions.csv").read_text(encoding="utf-8"),
                "report_md": (Path(output) / "report.md").read_text(encoding="utf-8")
                if (Path(output) / "report.md").is_file()
                else "",
            },
            "stdout": completed.stdout[-2000:],
        }


class BenchmarkSuiteRequest(BaseModel):
    datasets: list[Literal["t301-reconstructed", "t301-synthetic", "dwsim-synthetic"]] = Field(
        default_factory=lambda: ["t301-reconstructed"]
    )
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=6)
    graph_hops: list[int] = Field(default_factory=lambda: [2, 4], min_length=1, max_length=3)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    with_llm: bool = False
    max_llm_cases: int = Field(default=15, ge=0, le=100)


def _execute_benchmark(
    dataset: str, seed: int, test_fraction: float, with_llm: bool,
    max_llm_cases: int, max_hops: int, output: Path,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--dataset", dataset, "--seed", str(seed),
        "--test-fraction", str(test_fraction), "--max-hops", str(max_hops), "--output-dir", str(output),
    ]
    if with_llm:
        command.extend(["--with-llm", "--max-llm-cases", str(max_llm_cases)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SERVER_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=SERVER_ROOT, env=environment,
        capture_output=True, text=True, timeout=300, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "Benchmark failed.").strip()[-6000:])
    results_path = output / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset, "seed": seed, "max_hops": max_hops, "status": "completed",
        "metadata": results.get("metadata", {}),
        "method_summary": results.get("method_summary", {}),
        "case_count": len(results.get("cases", [])),
        "artifacts": {
            "results_json": "results.json",
            "summary_csv": "summary.csv",
            "predictions_csv": "predictions.csv",
            "report_md": "report.md",
        },
    }


@router.post("/benchmark-suite", summary="Run a reproducible multi-dataset, multi-seed benchmark suite")
def run_benchmark_suite(payload: BenchmarkSuiteRequest) -> dict[str, object]:
    """Execute a bounded matrix of real benchmark runs and persist immutable artifacts."""
    if not BENCHMARK_SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="Benchmark script is not available.")
    if len(set(payload.datasets)) != len(payload.datasets):
        raise HTTPException(status_code=422, detail="Dataset selections must be unique.")
    if len(set(payload.seeds)) != len(payload.seeds) or any(seed < 0 or seed > 1_000_000 for seed in payload.seeds):
        raise HTTPException(status_code=422, detail="Seeds must be unique integers between 0 and 1000000.")
    if len(set(payload.graph_hops)) != len(payload.graph_hops) or any(hops < 2 or hops > 8 for hops in payload.graph_hops):
        raise HTTPException(status_code=422, detail="Graph hop values must be unique integers between 2 and 8.")
    if len(payload.datasets) * len(payload.seeds) * len(payload.graph_hops) > 18:
        raise HTTPException(status_code=422, detail="A suite is limited to 18 dataset × seed × graph-hop runs.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    completed_runs: list[dict[str, object]] = []
    failures: list[dict[str, str | int]] = []
    for dataset in payload.datasets:
        for seed in payload.seeds:
            for max_hops in payload.graph_hops:
                output = run_dir / dataset / f"seed-{seed}" / f"hops-{max_hops}"
                try:
                    completed_runs.append(_execute_benchmark(
                        dataset, seed, payload.test_fraction, payload.with_llm,
                        payload.max_llm_cases, max_hops, output,
                    ))
                except subprocess.TimeoutExpired:
                    failures.append({"dataset": dataset, "seed": seed, "max_hops": max_hops, "error": "Benchmark timed out after 300 seconds."})
                except (OSError, RuntimeError, json.JSONDecodeError) as exc:
                    failures.append({"dataset": dataset, "seed": seed, "max_hops": max_hops, "error": str(exc)})
    # Aggregate independent seed-level results; one seed cannot estimate uncertainty.
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    grouped = defaultdict(lambda: defaultdict(list))
    for run in completed_runs:
        for method, metrics in run.get("method_summary", {}).items():
            for metric in ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "coverage", "abstention_rate", "mean_latency_ms"):
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    grouped[(str(run["dataset"]), int(run["max_hops"]), method)][metric].append(float(value))
    aggregate_summary = {}
    for (dataset, hops, method), metric_values in grouped.items():
        key = f"{dataset}|hops={hops}|{method}"
        aggregate_summary[key] = {"dataset": dataset, "max_hops": hops, "method": method, "metrics": {}}
        for metric, values in metric_values.items():
            count = len(values)
            mean = sum(values) / count
            sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1)) if count > 1 else None
            critical = t_critical.get(count, 1.96)
            margin = critical * sd / math.sqrt(count) if sd is not None else None
            lower = mean - margin if margin is not None else None
            upper = mean + margin if margin is not None else None
            if metric != "mean_latency_ms" and lower is not None:
                lower = max(0.0, lower)
                upper = min(1.0, upper)
            aggregate_summary[key]["metrics"][metric] = {
                "n_seeds": count, "mean": mean, "sample_sd": sd,
                "ci95_low": lower, "ci95_high": upper,
                "interval_method": "Student-t 95% CI" if count > 1 and count <= 6 else ("normal-approximation 95% CI" if count > 6 else "not estimable with one seed"),
            }
    manifest = {
        "run_id": run_id, "created_at": datetime.now(UTC).isoformat(),
        "protocol_version": "1.0", "label_status": "silver_row_association",
        "label_warning": "Co-listed row associations are not expert-validated causal ground truth.",
        "datasets": payload.datasets, "seeds": payload.seeds, "graph_hops": payload.graph_hops,
        "test_fraction": payload.test_fraction, "with_llm": payload.with_llm,
        "runs": completed_runs, "failures": failures, "aggregate_summary": aggregate_summary,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@router.get("/benchmark-runs", summary="List persisted benchmark suite runs")
def list_benchmark_runs() -> dict[str, object]:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    runs = []
    for manifest_path in sorted(RUNS_ROOT.glob("*/manifest.json"), reverse=True):
        try:
            runs.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return {"count": len(runs), "runs": runs}


@router.get("/benchmark-runs/{run_id}/{dataset}/{seed}/{max_hops}/{artifact}")
def get_benchmark_artifact(run_id: str, dataset: str, seed: int, max_hops: int, artifact: str) -> dict[str, str]:
    allowed = {"results.json", "summary.csv", "predictions.csv", "report.md"}
    if artifact not in allowed or Path(run_id).name != run_id or Path(dataset).name != dataset:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = RUNS_ROOT / run_id / dataset / f"seed-{seed}" / f"hops-{max_hops}" / artifact
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return {"filename": artifact, "content": path.read_text(encoding="utf-8")}
