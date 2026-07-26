"""Model-facing weather schema and leakage-safe cohort helpers."""
from __future__ import annotations

import pandas as pd

WEATHER_RAW_FEATURES: list[str] = [
    f"{prefix}{name}"
    for prefix in ("origin_", "destination_")
    for name in (
        "temperature_c",
        "dew_point_c",
        "visibility_m",
        "wind_speed_mps",
        "wind_gust_mps",
        "ceiling_m",
        "precipitation_mm",
        "thunderstorm_flag",
        "snow_flag",
        "fog_flag",
        "observation_age_minutes",
    )
]

WEATHER_DERIVED_FEATURES: list[str] = [
    f"{prefix}{name}"
    for prefix in ("origin_", "destination_")
    for name in (
        "low_visibility",
        "strong_wind",
        "low_ceiling",
        "active_precipitation",
        "freezing_conditions",
        "weather_severity",
    )
]

WEATHER_AVAILABILITY_FEATURES: list[str] = [
    "origin_weather_available",
    "destination_weather_available",
    "origin_weather_stale",
    "destination_weather_stale",
]

WEATHER_MODEL_FEATURES: list[str] = (
    WEATHER_RAW_FEATURES + WEATHER_DERIVED_FEATURES + WEATHER_AVAILABILITY_FEATURES
)


def complete_weather_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows with point-in-time weather available at both endpoints."""
    required = {"origin_weather_available", "destination_weather_available"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing weather availability columns: {sorted(missing)}")
    return (
        pd.to_numeric(frame["origin_weather_available"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(frame["destination_weather_available"], errors="coerce").fillna(0).eq(1)
    )


def impute_weather_from_training(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Median-impute numeric weather values using training data only.

    Binary/derived flags naturally receive a zero fallback when an entire
    training column is absent. The fitted medians never consult future rows.
    """
    selected = list(columns or WEATHER_MODEL_FEATURES)
    train_out = train.copy()
    other_out = [frame.copy() for frame in others]
    medians: dict[str, float] = {}
    for column in selected:
        train_values = pd.to_numeric(train_out[column], errors="coerce")
        median = train_values.median()
        medians[column] = 0.0 if pd.isna(median) else float(median)
        train_out[column] = train_values.fillna(medians[column])
        for frame in other_out:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(medians[column])
    return (train_out, *other_out)
