from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.monitoring import drift_summary, prediction_summary, psi, save_drift_reference
from src.monitoring.prediction_logger import log_prediction


class DummyPayload:
    airline = "DL"
    origin = "JFK"
    destination = "LAX"
    month = 7
    day_of_week = 5
    crs_dep_time = 1830
    crs_arr_time = 2145
    crs_elapsed_time = 375
    distance = 2475


def test_psi_is_zero_for_identical_distributions():
    values = np.array([1, 2, 3, 4, 5] * 20)
    assert psi(values, values) < 1e-9


def test_prediction_logger_and_summary(tmp_path: Path):
    path = tmp_path / "prediction_log.csv"
    log_prediction(
        DummyPayload(),
        {"delay_probability": 0.33, "risk_level": "moderate", "decision_threshold": 0.41},
        {"version": "4.0.0", "model_name": "extra_trees"},
        path=path,
    )
    summary = prediction_summary(path)
    assert summary["total_predictions"] == 1
    assert summary["model_name"] == "extra_trees"
    assert 0.0 <= summary["average_probability"] <= 1.0


def test_drift_summary_with_reference(tmp_path: Path):
    log_path = tmp_path / "prediction_log.csv"
    ref_path = tmp_path / "drift_reference.json"
    pd.DataFrame(
        [
            {
                "month": 7,
                "day_of_week": 5,
                "crs_elapsed_time": 300,
                "distance": 2000,
                "delay_probability": 0.3,
                "risk_level": "moderate",
                "timestamp_utc": "2026-01-01T00:00:00Z",
            }
        ]
    ).to_csv(log_path, index=False)
    save_drift_reference(
        {"numeric_features": {"Month": [1, 2, 3, 7], "DayOfWeek": [1, 2, 3, 5], "Distance": [500, 1000, 2000]}},
        ref_path,
    )
    result = drift_summary(log_path, ref_path)
    assert result["status"] in {"low", "moderate", "high"}
    assert "features" in result
