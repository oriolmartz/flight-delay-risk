"""Build the FlightRisk release manifest with hashes and live lineage metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.models.registry import FlightRiskArtifact
from src.version import APP_VERSION, RELEASE_NAME

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "RELEASE_MANIFEST.json"
TRACKED = [
    "models/flightrisk_model.joblib",
    "data/processed/data_manifest.json",
    "data/processed/schedule_context.joblib",
    "reports/metrics.json",
    "reports/temporal_backtest.json",
    "reports/calibration_report.json",
    "reports/candidate_benchmark.json",
    "reports/performance_benchmark.json",
    "reports/ui_smoke.json",
    "reports/neural_smoke.json",
    "reports/feature_ablation.json",
    "reports/feature_stability.json",
    "reports/operational_policy.json",
    "reports/policy_backtest.json",
    "reports/robustness_audit.json",
    "reports/drift_analysis.json",
    "reports/scale_refit.json",
    "reports/production_smoke.json",
    "docs/openapi.json",
    "reports/test_results.json",
    "LICENSE",
    "README.md",
    "README_ES.md",
    "requirements.txt",
    "requirements-advanced.txt",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    files = {}
    for relative in TRACKED:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}

    artifact = FlightRiskArtifact.load(ROOT / "models" / "flightrisk_model.joblib")
    backtest = _load_json("reports/temporal_backtest.json")
    calibration = _load_json("reports/calibration_report.json")
    policy = _load_json("reports/operational_policy.json")
    policy_backtest = _load_json("reports/policy_backtest.json")
    robustness = _load_json("reports/robustness_audit.json")
    drift = _load_json("reports/drift_analysis.json")
    stability = _load_json("reports/feature_stability.json")
    data_manifest = _load_json("data/processed/data_manifest.json")
    if artifact.metadata.get("data_sha256") != data_manifest.get("output_sha256"):
        raise RuntimeError("Artifact data fingerprint does not match the canonical data manifest")

    manifest = {
        "name": "Flight Delay Risk",
        "version": APP_VERSION,
        "release_name": RELEASE_NAME,
        "status": "stable_public_portfolio_release",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_schema_version": artifact.metadata.get("artifact_schema_version"),
        "interfaces": ["streamlit_en_es", "fastapi", "pdf_reports_en_es"],
        "quality": {
            "canonical_dataset_rows": data_manifest.get("dataset", {}).get("rows"),
            "complete_calendar_coverage": data_manifest.get("dataset", {}).get(
                "all_selected_months_complete"
            ),
            "data_sha256": data_manifest.get("output_sha256"),
            "training_protocol": artifact.metadata.get("training_protocol"),
            "selected_model": artifact.metadata.get("selected_model_key"),
            "temporal_backtest_folds": backtest.get("summary", {}).get("folds"),
            "backtest_selected_models": backtest.get("summary", {}).get(
                "selected_models", {}
            ),
            "calibration": calibration.get("method"),
            "calibration_protocol": calibration.get("protocol"),
            "historical_encoding": artifact.metadata.get("historical_encoding"),
            "explanation_method": artifact.metadata.get("explanation_method"),
            "operational_policy": policy.get("selected_policy"),
            "policy_backtest_mean_lift": policy_backtest.get("summary", {}).get("lift", {}).get("mean"),
            "weekly_block_bootstrap_samples": robustness.get("bootstrap_samples"),
            "paired_pr_auc_advantage_excludes_zero": robustness.get("paired_difference_vs_baseline", {}).get("pr_auc", {}).get("excludes_zero"),
            "drift_status": drift.get("feature_drift", {}).get("status"),
            "stable_feature_families": stability.get("selected_families"),
        },
        "files": files,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
