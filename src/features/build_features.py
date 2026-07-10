"""
Pre-flight feature engineering.

Every feature built here is derived ONLY from information known before
the flight departs (schedule, calendar, carrier, airports, distance, and
historical delay-rate statistics computed from training data). No
post-flight / actual-operation columns are used -- see
``src.config.FORBIDDEN_LEAKAGE_COLUMNS`` and the tests in
``tests/test_features.py`` which assert this explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FORBIDDEN_LEAKAGE_COLUMNS


def hhmm_to_hour(series: pd.Series) -> pd.Series:
    """Convert BTS HHMM-formatted scheduled time (e.g. 1830, 930, 5) to hour-of-day (0-23)."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    numeric = numeric.clip(lower=0, upper=2400)
    hour = (numeric // 100).astype(int) % 24
    return hour


def build_route(origin: pd.Series, dest: pd.Series) -> pd.Series:
    """Create a route identifier such as ``JFK_LAX``."""
    return origin.astype(str).str.upper() + "_" + dest.astype(str).str.upper()


def _distance_band(distance: pd.Series) -> pd.Series:
    distance = pd.to_numeric(distance, errors="coerce").fillna(0)
    bins = [-1, 250, 500, 1000, 2000, 4000, float("inf")]
    labels = ["very_short", "short", "medium", "long", "very_long", "ultra_long"]
    return pd.cut(distance, bins=bins, labels=labels).astype(str)


def _period_from_hour(hour: pd.Series) -> pd.Series:
    hour = pd.to_numeric(hour, errors="coerce").fillna(0).astype(int) % 24
    labels = pd.Series("midday", index=hour.index, dtype="object")
    labels[(hour >= 5) & (hour <= 8)] = "early_morning"
    labels[(hour >= 9) & (hour <= 11)] = "morning"
    labels[(hour >= 12) & (hour <= 15)] = "afternoon"
    labels[(hour >= 16) & (hour <= 19)] = "evening_peak"
    labels[(hour >= 20) & (hour <= 23)] = "night"
    labels[(hour >= 0) & (hour <= 4)] = "red_eye"
    return labels


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived pre-flight schedule/calendar features to ``df``.

    The feature layer adds richer schedule-only features. These are still leakage-safe:
    they use only scheduled times, route, carrier, distance and calendar fields.
    """
    df = df.copy()

    if "Airline" not in df.columns:
        raise KeyError("Expected an 'Airline' column (Reporting/Operating carrier code).")

    df["Airline"] = df["Airline"].astype(str).str.upper().str.strip()
    df["Origin"] = df["Origin"].astype(str).str.upper().str.strip()
    df["Dest"] = df["Dest"].astype(str).str.upper().str.strip()

    df["DepHour"] = hhmm_to_hour(df["CRSDepTime"])
    df["ArrHour"] = hhmm_to_hour(df["CRSArrTime"])
    df["Route"] = build_route(df["Origin"], df["Dest"])
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    df["DepPeriod"] = _period_from_hour(df["DepHour"])
    df["ArrPeriod"] = _period_from_hour(df["ArrHour"])
    df["DistanceBand"] = _distance_band(df["Distance"])

    # Operationally meaningful schedule flags known before departure.
    df["IsMorningPeak"] = df["DepHour"].between(6, 9).astype(int)
    df["IsEveningPeak"] = df["DepHour"].between(16, 19).astype(int)
    df["IsPeakHour"] = ((df["IsMorningPeak"] == 1) | (df["IsEveningPeak"] == 1)).astype(int)
    df["IsRedEye"] = ((df["DepHour"] <= 5) | (df["DepHour"] >= 22)).astype(int)
    df["IsLongHaul"] = (pd.to_numeric(df["Distance"], errors="coerce").fillna(0) >= 2000).astype(int)

    elapsed = pd.to_numeric(df["CRSElapsedTime"], errors="coerce").replace(0, np.nan)
    distance = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    df["ScheduledSpeedMph"] = (distance / (elapsed / 60.0)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["LogDistance"] = np.log1p(distance.clip(lower=0))
    df["DepHourSin"] = np.sin(2 * np.pi * df["DepHour"] / 24.0)
    df["DepHourCos"] = np.cos(2 * np.pi * df["DepHour"] / 24.0)
    df["ArrHourSin"] = np.sin(2 * np.pi * df["ArrHour"] / 24.0)
    df["ArrHourCos"] = np.cos(2 * np.pi * df["ArrHour"] / 24.0)

    # Composite keys used only to map train-fitted historical aggregates.
    df["CarrierRoute"] = df["Airline"] + "_" + df["Route"]
    df["AirlineOrigin"] = df["Airline"] + "_" + df["Origin"]
    df["AirlineDest"] = df["Airline"] + "_" + df["Dest"]
    df["OriginDepHour"] = df["Origin"] + "_" + df["DepHour"].astype(str)
    df["DestArrHour"] = df["Dest"] + "_" + df["ArrHour"].astype(str)
    df["CarrierDepHour"] = df["Airline"] + "_" + df["DepHour"].astype(str)

    return df


def assert_no_leakage_columns(feature_columns: list[str]) -> None:
    """Guard rail: raise if any forbidden post-flight column made it into the feature set."""
    leaked = set(feature_columns) & set(FORBIDDEN_LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(
            f"Leakage detected! The following forbidden columns are present in the "
            f"feature set: {sorted(leaked)}"
        )
