"""Run the local FlightRisk release quality gate."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.services import prediction_service
from src.models.predict import PredictionInput
from src.version import APP_VERSION

ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, command: list[str]) -> None:
    print(f"[gate] {label}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def _sample() -> PredictionInput:
    return PredictionInput(
        airline="DL",
        origin="JFK",
        destination="LAX",
        month=7,
        day_of_week=5,
        crs_dep_time=1830,
        crs_arr_time=2145,
        crs_elapsed_time=375,
        distance=2475,
    )


def main() -> int:
    print(f"FlightRisk v{APP_VERSION} quality gate")
    _run("compile", [sys.executable, "-m", "compileall", "-q", "app", "src", "scripts", "tests"])

    if shutil.which("ruff"):
        _run("ruff", ["ruff", "check", "app", "src", "scripts", "tests"])
    else:
        print("[gate] ruff: skipped (not installed)")

    _run("pytest", [sys.executable, "-m", "pytest", "-q"])

    prediction_service.get_artifact.cache_clear()
    if not prediction_service.is_model_available():
        raise RuntimeError("Model artifact is not available")

    sample = _sample()
    single = prediction_service.predict_flight(sample)
    batch = prediction_service.predict_flights_batch([sample, sample])
    if not 0 <= single["delay_probability"] <= 1:
        raise RuntimeError("Single prediction probability is outside [0, 1]")
    if len(batch) != 2:
        raise RuntimeError("Batch inference did not return two predictions")

    metrics_path = ROOT / "reports" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "main_model" not in metrics or "baseline_model" not in metrics:
        raise RuntimeError("reports/metrics.json is missing required model blocks")

    print("[gate] artifact load: passed")
    print("[gate] single inference: passed")
    print("[gate] vectorized batch inference: passed")
    print("[gate] committed reports: passed")
    print("[gate] FlightRisk quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
