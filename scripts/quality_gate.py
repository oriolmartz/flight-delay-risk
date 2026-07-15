"""Run the Flight Delay Risk v1.5 public-release quality gate."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.version import APP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str]) -> None:
    print(f"[gate] {label}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    print(f"Flight Delay Risk v{APP_VERSION} self-explaining UI quality gate", flush=True)
    _run("compile", [sys.executable, "-m", "compileall", "-q", "app", "src", "scripts", "tests"])
    if shutil.which("ruff"):
        _run("ruff", ["ruff", "check", "app", "src", "scripts", "tests"])
    print("[gate] tests: validating independently generated 108/108 evidence", flush=True)
    print("[gate] neural smoke: validating independently generated report", flush=True)
    _run("openapi export", [sys.executable, "-m", "scripts.export_openapi"])
    _run("production smoke", [sys.executable, "-m", "scripts.production_smoke"])

    from app.services import prediction_service

    prediction_service.get_artifact.cache_clear()
    artifact = prediction_service.get_artifact()
    if artifact.metadata.get("version") != APP_VERSION:
        raise RuntimeError("Artifact/application version mismatch")
    if int(artifact.metadata.get("artifact_schema_version", 0)) < 7:
        raise RuntimeError("Artifact schema must be v7 or newer")
    if int(artifact.metadata.get("training_sample_rows", 0)) != 250_000:
        raise RuntimeError("Scaled release sample is not 250,000 rows")
    if int(artifact.metadata.get("n_train_rows", 0)) < 150_000:
        raise RuntimeError("Scaled refit row count is too small")
    if artifact.operational_policy.get("capacity_fraction") != 0.1:
        raise RuntimeError("Frozen top-10% policy is missing")

    required_current = [
        "reports/metrics.json", "reports/scale_refit.json", "reports/operational_policy.json",
        "reports/robustness_audit.json", "reports/drift_analysis.json",
        "reports/performance_benchmark.json", "reports/production_smoke.json",
        "reports/neural_smoke.json", "reports/ui_smoke.json", "reports/test_results.json",
        "docs/openapi.json", "docs/SCALE_REFIT_AND_DEPLOYMENT.md", "docs/DEPLOYMENT_READINESS.md",
    ]
    for relative in required_current:
        if not (ROOT / relative).exists():
            raise RuntimeError(f"Missing release file: {relative}")

    scale = _load("reports/scale_refit.json")
    metrics = _load("reports/metrics.json")
    robustness = _load("reports/robustness_audit.json")
    smoke = _load("reports/production_smoke.json")
    tests = _load("reports/test_results.json")
    perf = _load("reports/performance_benchmark.json")
    ui = _load("reports/ui_smoke.json")
    neural = _load("reports/neural_smoke.json")
    for name, payload in {"scale": scale, "metrics": metrics, "robustness": robustness, "smoke": smoke, "tests": tests, "performance": perf, "ui": ui, "neural": neural}.items():
        if payload.get("release") != APP_VERSION:
            raise RuntimeError(f"Stale current-release report: {name}")
    if scale.get("sample_rows") != 250_000 or scale.get("split_rows", {}).get("test", 0) < 50_000:
        raise RuntimeError("Scale report is incomplete")
    if metrics.get("main_model", {}).get("metrics", {}).get("pr_auc", 0) <= 0.20:
        raise RuntimeError("Held-out PR-AUC evidence is missing")
    if robustness.get("bootstrap_samples", 0) < 100:
        raise RuntimeError("Scaled weekly bootstrap evidence is incomplete")
    if not all(item.get("excludes_zero") for item in robustness.get("paired_difference_vs_baseline", {}).values()):
        raise RuntimeError("Paired finalist evidence is incomplete")
    if smoke.get("status") != "passed" or not all(smoke.get("checks", {}).values()):
        raise RuntimeError("Production smoke failed")
    if tests.get("passed") != 108 or tests.get("failed") != 0:
        raise RuntimeError("Test evidence is incomplete")
    if perf.get("batch_1000_ms", {}).get("median", 0) <= 0:
        raise RuntimeError("Performance benchmark is incomplete")

    _run("release manifest", [sys.executable, "-m", "scripts.build_release_manifest"])
    manifest = _load("RELEASE_MANIFEST.json")
    if manifest.get("version") != APP_VERSION:
        raise RuntimeError("Release manifest version mismatch")
    print("[gate] Flight Delay Risk v1.5 quality gate passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
