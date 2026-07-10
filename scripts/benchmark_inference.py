"""Measure release artifact load and inference latency."""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

from src.config import DEFAULT_MODEL_PATH
from src.models.predict import PredictionInput, predict_batch, predict_single
from src.models.registry import FlightRiskArtifact
from src.version import APP_VERSION

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "reports" / "performance_benchmark.json"


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


def _measure(callable_obj, repeats: int = 7) -> dict[str, float]:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_obj()
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "median": round(statistics.median(durations), 3),
        "min": round(min(durations), 3),
        "max": round(max(durations), 3),
        "repeats": repeats,
    }


def main() -> int:
    load_start = time.perf_counter()
    artifact = FlightRiskArtifact.load(DEFAULT_MODEL_PATH)
    artifact_load_ms = (time.perf_counter() - load_start) * 1000
    sample = _sample()

    # Warm preprocessing and model kernels before timing steady-state inference.
    predict_single(artifact, sample, artifact.decision_threshold)
    predict_batch(artifact, [sample] * 10, artifact.decision_threshold)

    report = {
        "release": APP_VERSION,
        "artifact_load_ms": round(artifact_load_ms, 3),
        "single_prediction_ms": _measure(
            lambda: predict_single(artifact, sample, artifact.decision_threshold), repeats=15
        ),
        "batch_100_ms": _measure(
            lambda: predict_batch(artifact, [sample] * 100, artifact.decision_threshold), repeats=7
        ),
        "batch_1000_ms": _measure(
            lambda: predict_batch(artifact, [sample] * 1000, artifact.decision_threshold), repeats=5
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "implementation": platform.python_implementation(),
        },
        "note": "Steady-state local measurements after warm-up; cloud hosting adds network and cold-start latency.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
