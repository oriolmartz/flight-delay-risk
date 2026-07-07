"""Monitoring summaries and lightweight drift checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DRIFT_REPORT_PATH, PREDICTION_LOG_PATH


def prediction_summary(path: Path = PREDICTION_LOG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "total_predictions": 0,
            "average_probability": None,
            "high_risk_share": None,
            "latest_prediction_utc": None,
            "model_name": None,
            "model_version": None,
        }
    df = pd.read_csv(path)
    if df.empty:
        return {"total_predictions": 0}
    return {
        "total_predictions": int(len(df)),
        "average_probability": float(df["delay_probability"].mean()),
        "high_risk_share": float((df["risk_level"] == "high").mean()),
        "moderate_or_high_risk_share": float(df["risk_level"].isin(["moderate", "high"]).mean()),
        "latest_prediction_utc": str(df["timestamp_utc"].iloc[-1]),
        "model_name": str(df.get("model_name", pd.Series(["unknown"])).iloc[-1]),
        "model_version": str(df.get("model_version", pd.Series(["unknown"])).iloc[-1]),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        min_v = min(expected.min(), actual.min())
        max_v = max(expected.max(), actual.max())
        if min_v == max_v:
            return 0.0
        quantiles = np.linspace(min_v, max_v, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.clip(expected_counts / max(1, expected_counts.sum()), 1e-6, 1)
    actual_pct = np.clip(actual_counts / max(1, actual_counts.sum()), 1e-6, 1)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def build_drift_reference(X_train: pd.DataFrame) -> dict[str, Any]:
    numeric_features = ["Month", "DayOfWeek", "DepHour", "ArrHour", "CRSElapsedTime", "Distance"]
    ref: dict[str, Any] = {"numeric_features": {}}
    for col in numeric_features:
        if col in X_train.columns:
            ref["numeric_features"][col] = X_train[col].astype(float).dropna().tolist()
    return ref


def save_drift_reference(reference: dict[str, Any], path: Path = DRIFT_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store compact samples to keep repo artifacts small.
    compact = {"numeric_features": {}}
    for col, values in reference.get("numeric_features", {}).items():
        series = pd.Series(values).dropna()
        if len(series) > 1000:
            series = series.sample(1000, random_state=42)
        compact["numeric_features"][col] = series.astype(float).round(4).tolist()
    path.write_text(json.dumps(compact, indent=2))


def drift_summary(
    log_path: Path = PREDICTION_LOG_PATH,
    reference_path: Path = DRIFT_REPORT_PATH,
) -> dict[str, Any]:
    if not log_path.exists():
        return {"status": "no_prediction_log", "features": {}}
    if not reference_path.exists():
        return {"status": "no_reference", "features": {}}
    log_df = pd.read_csv(log_path)
    ref = json.loads(reference_path.read_text())
    features: dict[str, float] = {}
    mapping = {
        "Month": "month",
        "DayOfWeek": "day_of_week",
        "CRSElapsedTime": "crs_elapsed_time",
        "Distance": "distance",
    }
    for ref_col, log_col in mapping.items():
        if ref_col in ref.get("numeric_features", {}) and log_col in log_df.columns:
            features[ref_col] = psi(np.array(ref["numeric_features"][ref_col]), log_df[log_col].to_numpy())
    max_psi = max(features.values(), default=0.0)
    if max_psi >= 0.25:
        status = "high"
    elif max_psi >= 0.1:
        status = "moderate"
    else:
        status = "low"
    return {"status": status, "max_psi": max_psi, "features": features}
