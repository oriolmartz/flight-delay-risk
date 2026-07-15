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
PREDICTION_LOG_PATH: Path = Path(os.getenv("FLIGHTRISK_PREDICTION_LOG_PATH", MONITORING_DIR / "prediction_log.csv"))
DRIFT_REPORT_PATH: Path = REPORTS_DIR / "drift_reference.json"
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"

DEFAULT_PROCESSED_PATH: Path = PROCESSED_DATA_DIR / "flights_processed.parquet"
DATA_MANIFEST_PATH: Path = PROCESSED_DATA_DIR / "data_manifest.json"
SCHEDULE_CONTEXT_PATH: Path = Path(os.getenv("FLIGHTRISK_SCHEDULE_CONTEXT_PATH", PROCESSED_DATA_DIR / "schedule_context.joblib"))
DEFAULT_MODEL_PATH: Path = Path(os.getenv("FLIGHTRISK_MODEL_PATH", MODELS_DIR / "flightrisk_model.joblib"))
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
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    TARGET_COL,
]


# Canonical columns persisted in the processed training dataset. Keeping the
# processed schema narrow makes chunked preparation deterministic across BTS
# exports that contain different optional columns.
PROCESSED_COLUMNS: list[str] = [
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    TARGET_COL,
]

# Final leakage-safe feature schema grouped for auditable ablation.
CORE_SCHEDULE_FEATURES: list[str] = [
    "Airline", "Origin", "Dest", "Route", "DepPeriod", "ArrPeriod", "DistanceBand",
    "Month", "DayOfWeek", "IsWeekend", "DepHour", "ArrHour", "DepMinute", "ArrMinute",
    "CRSElapsedTime", "Distance", "ScheduledSpeedMph", "LogDistance",
    "DepHourSin", "DepHourCos", "ArrHourSin", "ArrHourCos",
    "DepMinuteSin", "DepMinuteCos", "ArrMinuteSin", "ArrMinuteCos",
    "IsMorningPeak", "IsEveningPeak", "IsPeakHour", "IsRedEye", "IsLongHaul",
    "IsOvernightSchedule",
]
CALENDAR_FEATURES: list[str] = [
    "Season", "CalendarDateKnown", "DayOfMonth", "DayOfYear", "WeekOfYear", "Quarter",
    "YearProgress", "MonthSin", "MonthCos", "DayOfWeekSin", "DayOfWeekCos",
    "YearDaySin", "YearDayCos", "DaysToNearestFederalHoliday", "IsFederalHoliday",
    "IsHolidayWindow",
]
HISTORICAL_RATE_FEATURES: list[str] = [
    "CarrierDelayRate", "RouteDelayRate", "OriginDelayRate", "DestDelayRate",
    "CarrierRouteDelayRate", "AirlineOriginDelayRate", "AirlineDestDelayRate",
    "OriginHourDelayRate", "DestHourDelayRate", "CarrierDepHourDelayRate",
    "RouteFlightShare", "CarrierRouteFlightShare", "AirlineOriginFlightShare",
    "OriginHourFlightShare", "DestHourFlightShare", "CarrierDepHourFlightShare",
]
HISTORICAL_SUPPORT_FEATURES: list[str] = [
    f"{prefix}{suffix}"
    for prefix in (
        "CarrierHistory", "RouteHistory", "OriginHistory", "DestHistory",
        "CarrierRouteHistory", "AirlineOriginHistory", "AirlineDestHistory",
        "OriginHourHistory", "DestHourHistory", "CarrierDepHourHistory",
    )
    for suffix in ("Count", "LogCount")
]
RECENCY_FEATURES: list[str] = [
    f"{prefix}{suffix}"
    for prefix in ("CarrierDelay", "RouteDelay", "OriginDelay", "DestDelay")
    for suffix in ("Rate28d", "Rate90d", "RateEWMA", "Trend28d")
]
SCHEDULE_CONGESTION_FEATURES: list[str] = [
    "OriginScheduledDepartures30m", "OriginScheduledDepartures60m",
    "OriginScheduledDepartures120m", "DestScheduledArrivals30m",
    "DestScheduledArrivals60m", "DestScheduledArrivals120m",
    "OriginDailyScheduledFlights", "DestDailyScheduledFlights",
    "CarrierOriginDailyScheduledFlights", "RouteDailyScheduledFlights",
    "OriginBankShare60m", "DestBankShare60m",
]
FEATURE_FAMILIES: dict[str, list[str]] = {
    "core_schedule": CORE_SCHEDULE_FEATURES,
    "calendar": CALENDAR_FEATURES,
    "historical_rates": HISTORICAL_RATE_FEATURES,
    "historical_support": HISTORICAL_SUPPORT_FEATURES,
    "recency": RECENCY_FEATURES,
    "schedule_congestion": SCHEDULE_CONGESTION_FEATURES,
}
FEATURE_COLUMNS: list[str] = [
    column for family in FEATURE_FAMILIES.values() for column in family
]
CATEGORICAL_FEATURES: list[str] = [
    "Airline", "Origin", "Dest", "Route", "DepPeriod", "ArrPeriod", "DistanceBand", "Season"
]
NUMERIC_FEATURES: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# Global fallback value used for historical delay-rate features when a
# carrier / route / airport has never been seen in the training data.
GLOBAL_FALLBACK_DELAY_RATE: float = 0.2
