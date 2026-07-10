"""Run the FlightRisk v1.0 public-release quality gate."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.services import prediction_service, report_service
from src.config import PREDICTION_LOG_PATH
from src.models.predict import PredictionInput, predict_batch, predict_single
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
    print(f"FlightRisk v{APP_VERSION} public-release quality gate")
    log_backup = PREDICTION_LOG_PATH.read_bytes() if PREDICTION_LOG_PATH.exists() else None
    try:
        _run("compile", [sys.executable, "-m", "compileall", "-q", "app", "src", "scripts", "tests"])
        if shutil.which("ruff"):
            _run("ruff", ["ruff", "check", "app", "src", "scripts", "tests"])
        else:
            print("[gate] ruff: skipped (not installed)")
        _run("pytest", [sys.executable, "-m", "pytest", "-q"])

        prediction_service.get_artifact.cache_clear()
        artifact = prediction_service.get_artifact()
        sample = _sample()
        single = predict_single(artifact, sample, artifact.decision_threshold)
        batch = predict_batch(artifact, [sample, sample], artifact.decision_threshold)
        context = prediction_service.prediction_context(sample)

        if not 0 <= single["delay_probability"] <= 1:
            raise RuntimeError("Single prediction probability is outside [0, 1]")
        if single.get("calibration_method") == "identity":
            raise RuntimeError("Release artifact is missing post-hoc calibration")
        if not single.get("local_contributions"):
            raise RuntimeError("Release artifact does not expose model-native contributions")
        if single.get("explanation_scale") != "log_odds_before_calibration":
            raise RuntimeError("Unexpected explanation scale")
        if len(batch) != 2:
            raise RuntimeError("Batch inference did not return two predictions")

        flight_meta = {
            "airline": "DL",
            "flight_number": "418",
            "origin": "JFK",
            "destination": "LAX",
            "scheduled_departure": "18:30",
            "scheduled_arrival": "21:45",
            "review_label": "WATCH",
        }
        for language in ("en", "es"):
            pdf = report_service.build_flight_brief_pdf(flight_meta, single, context, lang=language)
            if not pdf.startswith(b"%PDF") or len(pdf) < 2000:
                raise RuntimeError(f"Invalid {language} flight PDF")

        if artifact.metadata.get("version") != APP_VERSION:
            raise RuntimeError("Artifact version does not match the application release")
        if artifact.metadata.get("artifact_schema_version") != "2":
            raise RuntimeError("Expected artifact schema version 2")
        if artifact.metadata.get("historical_encoding") != "strictly_prior_flight_date":
            raise RuntimeError("Release artifact is missing ordered historical encoding metadata")
        if artifact.metadata.get("explanation_method") != "signed_linear_log_odds_contributions":
            raise RuntimeError("Release artifact is missing explanation metadata")

        required_reports = [
            "metrics.json",
            "temporal_backtest.json",
            "candidate_benchmark.json",
            "calibration_report.json",
            "performance_benchmark.json",
            "ui_smoke.json",
        ]
        for name in required_reports:
            if not (ROOT / "reports" / name).exists():
                raise RuntimeError(f"Required report is missing: {name}")

        metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
        main_metrics = metrics.get("main_model", {}).get("metrics", {})
        for metric in ("pr_auc", "lift_at_top_10pct", "brier_score", "expected_calibration_error"):
            if metric not in main_metrics:
                raise RuntimeError(f"reports/metrics.json is missing {metric}")

        backtest = json.loads((ROOT / "reports" / "temporal_backtest.json").read_text(encoding="utf-8"))
        if backtest.get("protocol", {}).get("release") != APP_VERSION:
            raise RuntimeError("Temporal backtest release does not match the application version")
        if backtest.get("summary", {}).get("folds", 0) < 3:
            raise RuntimeError("Temporal backtest report must contain at least three folds")

        ui_smoke = json.loads((ROOT / "reports" / "ui_smoke.json").read_text(encoding="utf-8"))
        if ui_smoke.get("release") != APP_VERSION or ui_smoke.get("status") != "passed":
            raise RuntimeError("Bilingual UI smoke report is missing or stale")

        performance = json.loads((ROOT / "reports" / "performance_benchmark.json").read_text(encoding="utf-8"))
        if performance.get("release") != APP_VERSION:
            raise RuntimeError("Performance benchmark version mismatch")
        if performance.get("batch_1000_ms", {}).get("median", 0) <= 0:
            raise RuntimeError("Performance benchmark is incomplete")

        for required_file in (
            ROOT / "README.md",
            ROOT / "README_ES.md",
            ROOT / "data" / "sample" / "schedule_template.csv",
            ROOT / "docs" / "PUBLIC_RELEASE.md",
        ):
            if not required_file.exists():
                raise RuntimeError(f"Public-release file is missing: {required_file.relative_to(ROOT)}")

        manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("version") != APP_VERSION:
            raise RuntimeError("Release manifest version does not match the application version")
        if not manifest.get("files", {}).get("models/flightrisk_model.joblib", {}).get("sha256"):
            raise RuntimeError("Release manifest is missing the artifact hash")

        print("[gate] artifact load: passed")
        print("[gate] calibrated inference: passed")
        print("[gate] model-native explanation: passed")
        print("[gate] bilingual PDF generation: passed")
        print("[gate] temporal and performance reports: passed")
        print("[gate] bilingual Streamlit workflow report: passed")
        print("[gate] FlightRisk public-release quality gate passed")
        return 0
    finally:
        if log_backup is None:
            PREDICTION_LOG_PATH.unlink(missing_ok=True)
        else:
            PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREDICTION_LOG_PATH.write_bytes(log_backup)


if __name__ == "__main__":
    raise SystemExit(main())
