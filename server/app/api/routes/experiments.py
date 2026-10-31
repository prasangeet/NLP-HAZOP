"""Research benchmark endpoints for the Experiments workbench."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/experiments", tags=["experiments"])
SERVER_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = SERVER_ROOT.parent
BENCHMARK_SCRIPT = SERVER_ROOT / "scripts" / "run_benchmark.py"


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
