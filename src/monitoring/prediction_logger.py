"""Post-deployment prediction logging for the portfolio API."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PREDICTION_LOG_PATH

LOG_COLUMNS = [
    "timestamp_utc",
    "model_version",
    "model_name",
    "airline",
    "origin",
    "destination",
    "month",
    "day_of_week",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
    "delay_probability",
    "risk_level",
    "decision_threshold",
]


def log_prediction(payload: Any, result: dict, metadata: dict, path: Path = PREDICTION_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("version", "unknown"),
        "model_name": metadata.get("model_name", "unknown"),
        "airline": payload.airline,
        "origin": payload.origin,
        "destination": payload.destination,
        "month": payload.month,
        "day_of_week": payload.day_of_week,
        "crs_dep_time": payload.crs_dep_time,
        "crs_arr_time": payload.crs_arr_time,
        "crs_elapsed_time": payload.crs_elapsed_time,
        "distance": payload.distance,
        "delay_probability": result["delay_probability"],
        "risk_level": result["risk_level"],
        "decision_threshold": result["decision_threshold"],
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
