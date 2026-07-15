"""Release-contract tests for Layer 5 scale refit and deployment readiness."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app
from app.services import prediction_service
from src.config import DEFAULT_MODEL_PATH
from src.models.registry import FlightRiskArtifact
from src.version import APP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_layer5_artifact_is_scaled_and_schema_v7():
    artifact = FlightRiskArtifact.load(DEFAULT_MODEL_PATH)
    assert artifact.metadata["version"] == APP_VERSION == "1.5.0"
    assert int(artifact.metadata["artifact_schema_version"]) >= 7
    assert artifact.metadata["training_sample_rows"] == 250_000
    assert artifact.metadata["n_train_rows"] >= 150_000
    assert artifact.metadata["scale_factor_vs_v1_3"] >= 8.0


def test_scale_refit_report_matches_artifact():
    report = json.loads((ROOT / "reports/scale_refit.json").read_text(encoding="utf-8"))
    assert report["release"] == APP_VERSION
    assert report["sample_rows"] == 250_000
    assert report["split_rows"]["test"] >= 50_000
    assert report["current_release"]["pr_auc"] > 0.20


def test_live_ready_and_trace_headers():
    prediction_service.get_artifact.cache_clear()
    client = TestClient(app)
    live = client.get("/live", headers={"x-request-id": "layer5-test"})
    assert live.status_code == 200
    assert live.json() == {"status": "alive", "version": APP_VERSION}
    assert live.headers["x-request-id"] == "layer5-test"
    assert live.headers["x-flightrisk-version"] == APP_VERSION

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["checks"]["schema_v7_or_newer"] is True


def test_deployment_descriptors_use_readiness_healthchecks():
    assert "/ready" in (ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "/ready" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "healthCheckPath: /ready" in (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "/_stcore/health" in (ROOT / "Dockerfile.dashboard").read_text(encoding="utf-8")


def test_exported_openapi_and_production_smoke_are_current():
    schema = json.loads((ROOT / "docs/openapi.json").read_text(encoding="utf-8"))
    smoke = json.loads((ROOT / "reports/production_smoke.json").read_text(encoding="utf-8"))
    assert schema["info"]["version"] == APP_VERSION
    assert "/ready" in schema["paths"]
    assert smoke["release"] == APP_VERSION
    assert smoke["status"] == "passed"
    assert all(smoke["checks"].values())
