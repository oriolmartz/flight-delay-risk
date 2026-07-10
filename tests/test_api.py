"""Tests for the FastAPI app: health check and predict endpoint with a sample payload.

A small model is trained on the bundled synthetic sample dataset and saved
to the default model path before these tests run, so the API has an
artifact to load (mirrors what `python -m scripts.run_local_demo` does).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import DEFAULT_MODEL_PATH, SAMPLE_CSV_PATH
from src.data.clean import clean_flights
from src.data.load_data import normalize_columns
from src.data.split import split_train_test
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.train import train_models


@pytest.fixture(scope="module", autouse=True)
def trained_model_artifact():
    """Train a small model on sample data and save it, then clear the service cache."""
    previous = os.environ.get("FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT")
    os.environ["FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT"] = "1"
    context_path = Path("data/europe/europe_punctuality_context.csv")
    sample_context_path = Path("data/europe/europe_punctuality_sample.csv")
    context_preexisting = context_path.exists()
    context_backup = context_path.read_bytes() if context_preexisting else None
    shutil.copyfile(sample_context_path, context_path)
    model_path = Path(DEFAULT_MODEL_PATH)
    model_preexisting = model_path.exists()
    model_backup = model_path.read_bytes() if model_preexisting else None

    raw_df = pd.read_csv(SAMPLE_CSV_PATH)
    raw_df = normalize_columns(raw_df)
    clean_df = clean_flights(raw_df)

    train_df, test_df = split_train_test(clean_df, test_size=0.25)
    models, aggregates, _, _ = train_models(train_df.copy())

    metadata = build_metadata(
        model_name=models["main"].name, n_train=len(train_df), n_test=len(test_df)
    )
    artifact = FlightRiskArtifact(
        pipeline=models["main"].pipeline,
        historical_aggregates=aggregates,
        metadata=metadata,
        metrics={"main_model": {"roc_auc": 0.5}, "baseline_model": {"roc_auc": 0.5}},
        decision_threshold=0.42,
    )
    artifact.save(DEFAULT_MODEL_PATH)

    # Ensure the service's cached artifact (if any) is refreshed.
    from app.services import prediction_service

    prediction_service.get_artifact.cache_clear()

    yield

    prediction_service.get_artifact.cache_clear()
    if model_preexisting and model_backup is not None:
        model_path.write_bytes(model_backup)
    else:
        model_path.unlink(missing_ok=True)
    if context_preexisting and context_backup is not None:
        context_path.write_bytes(context_backup)
    else:
        context_path.unlink(missing_ok=True)
    if previous is None:
        os.environ.pop("FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT", None)
    else:
        os.environ["FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT"] = previous


@pytest.fixture()
def client() -> TestClient:
    from app.api.main import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestModelInfoEndpoint:
    def test_model_info_returns_metadata(self, client: TestClient):
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert "model_name" in body
        assert "feature_columns" in body
        assert body["decision_threshold"] == 0.42


class TestModelCardEndpoint:
    def test_model_card_returns_leakage_controls(self, client: TestClient):
        response = client.get("/model/card")
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "FlightRisk"
        assert body["decision_threshold"] == 0.42
        assert len(body["leakage_controls"]) >= 3


class TestPredictEndpoint:
    SAMPLE_PAYLOAD = {
        "airline": "DL",
        "origin": "JFK",
        "destination": "LAX",
        "month": 7,
        "day_of_week": 5,
        "crs_dep_time": 1830,
        "crs_arr_time": 2145,
        "crs_elapsed_time": 375,
        "distance": 2475,
    }

    def test_predict_returns_valid_response(self, client: TestClient):
        response = client.post("/predict", json=self.SAMPLE_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["delay_probability"] <= 1.0
        assert body["risk_level"] in {"low", "moderate", "high"}
        assert isinstance(body["top_factors"], list)
        assert len(body["top_factors"]) > 0

    def test_predict_rejects_invalid_month(self, client: TestClient):
        bad_payload = {**self.SAMPLE_PAYLOAD, "month": 13}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_rejects_invalid_hhmm_minutes(self, client: TestClient):
        bad_payload = {**self.SAMPLE_PAYLOAD, "crs_dep_time": 1260}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_accepts_2400_midnight(self, client: TestClient):
        payload = {**self.SAMPLE_PAYLOAD, "crs_arr_time": 2400}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


    def test_predict_european_returns_valid_response(self, client: TestClient):
        payload = {
            "airline": "IB",
            "origin": "BCN",
            "destination": "AMS",
            "month": 7,
            "day_of_week": 5,
            "crs_dep_time": 845,
            "crs_arr_time": 1110,
            "crs_elapsed_time": 145
        }
        response = client.post("/predict/european", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["region"] == "europe_experimental"
        assert body["experimental"] is True
        assert body["distance_miles"] > 0
        assert "european_context" in body
        assert body["european_context"]["status"] in {"matched", "missing", "unavailable"}

    def test_regions_europe_catalog(self, client: TestClient):
        response = client.get("/regions/europe")
        assert response.status_code == 200
        body = response.json()
        assert body["region"] == "europe_experimental"
        assert len(body["airports"]) >= 5
        assert len(body["airlines"]) >= 5
        assert "context_summary" in body

    def test_regions_europe_context_summary(self, client: TestClient):
        response = client.get("/regions/europe/context")
        assert response.status_code == 200
        body = response.json()
        assert "available" in body
        assert "rows" in body

    def test_predict_batch(self, client: TestClient):
        response = client.post(
            "/predict/batch", json={"flights": [self.SAMPLE_PAYLOAD, self.SAMPLE_PAYLOAD]}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2

    def test_rank_endpoint_returns_sorted_predictions(self, client: TestClient):
        payload_2 = {**self.SAMPLE_PAYLOAD, "origin": "ATL", "destination": "BOS", "distance": 946, "crs_elapsed_time": 150}
        response = client.post("/rank", json={"flights": [self.SAMPLE_PAYLOAD, payload_2]})
        assert response.status_code == 200
        body = response.json()
        assert body["flights_ranked"] == 2
        assert len(body["ranked_predictions"]) == 2
        probs = [row["delay_probability"] for row in body["ranked_predictions"]]
        assert probs == sorted(probs, reverse=True)
        assert body["ranked_predictions"][0]["rank"] == 1


class TestMonitoringEndpoints:
    def test_monitoring_summary_returns_payload(self, client: TestClient):
        response = client.get("/monitoring/summary")
        assert response.status_code == 200
        body = response.json()
        assert "total_predictions" in body

    def test_drift_endpoint_returns_payload(self, client: TestClient):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "features" in body
