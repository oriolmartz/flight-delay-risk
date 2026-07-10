"""
Central configuration for the FlightRisk project.

All paths are resolved relative to the repository root so the project
runs the same way on Windows, macOS and Linux, and so no absolute
paths are hardcoded anywhere else in the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root = two levels up from this file (src/config.py -> src -> root)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"

MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"
MONITORING_DIR: Path = ROOT_DIR / "monitoring"
PREDICTION_LOG_PATH: Path = MONITORING_DIR / "prediction_log.csv"
DRIFT_REPORT_PATH: Path = REPORTS_DIR / "drift_reference.json"
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"

DEFAULT_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / "flights_processed.parquet"
DEFAULT_MODEL_PATH: Path = MODELS_DIR / "flightrisk_model.joblib"
SAMPLE_CSV_PATH: Path = SAMPLE_DATA_DIR / "sample_flights.csv"

# Target column
TARGET_COL: str = "ArrDel15"

# Random seed used everywhere for reproducibility
RANDOM_SEED: int = 42

# Decision threshold used to translate probability -> binary risk decision
DEFAULT_DECISION_THRESHOLD: float = float(os.getenv("FLIGHTRISK_THRESHOLD", "0.5"))

# Risk level cut points (probability of ArrDel15 == 1)
RISK_LOW_MAX: float = 0.25
RISK_MODERATE_MAX: float = 0.5

# Columns that are forbidden as model features because they leak
# post-flight / actual-operation information. A prediction must only ever
# be based on information known BEFORE the flight departs.
FORBIDDEN_LEAKAGE_COLUMNS: list[str] = [
    "ArrDelay",
    "ArrDelayMinutes",
    "DepDelay",
    "DepDelayMinutes",
    "ActualElapsedTime",
    "AirTime",
    "TaxiOut",
    "TaxiIn",
    "WheelsOff",
    "WheelsOn",
    "DepTime",
    "ArrTime",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    "Cancelled",
    "Diverted",
    "CancellationCode",
]

# Columns we require to exist (after normalization) in raw BTS CSVs.
REQUIRED_RAW_COLUMNS: list[str] = [
    "Year",
    "Month",
    "DayOfWeek",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    TARGET_COL,
]

# Final feature columns used by the model (pre-flight information only).
# The feature set includes richer schedule/context features while preserving leakage safety.
FEATURE_COLUMNS: list[str] = [
    "Airline",
    "Origin",
    "Dest",
    "Route",
    "DepPeriod",
    "ArrPeriod",
    "DistanceBand",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "DepHour",
    "ArrHour",
    "CRSElapsedTime",
    "Distance",
    "ScheduledSpeedMph",
    "LogDistance",
    "DepHourSin",
    "DepHourCos",
    "ArrHourSin",
    "ArrHourCos",
    "IsMorningPeak",
    "IsEveningPeak",
    "IsPeakHour",
    "IsRedEye",
    "IsLongHaul",
    "CarrierDelayRate",
    "RouteDelayRate",
    "OriginDelayRate",
    "DestDelayRate",
    "CarrierRouteDelayRate",
    "AirlineOriginDelayRate",
    "AirlineDestDelayRate",
    "OriginHourDelayRate",
    "DestHourDelayRate",
    "CarrierDepHourDelayRate",
    "RouteFlightShare",
    "CarrierRouteFlightShare",
    "AirlineOriginFlightShare",
    "OriginHourFlightShare",
    "DestHourFlightShare",
    "CarrierDepHourFlightShare",
]

CATEGORICAL_FEATURES: list[str] = [
    "Airline",
    "Origin",
    "Dest",
    "Route",
    "DepPeriod",
    "ArrPeriod",
    "DistanceBand",
]
NUMERIC_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# Global fallback value used for historical delay-rate features when a
# carrier / route / airport has never been seen in the training data.
GLOBAL_FALLBACK_DELAY_RATE: float = 0.2
