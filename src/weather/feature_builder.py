"""Derived operational weather features with explicit thresholds."""
from __future__ import annotations

import pandas as pd


def build_weather_features(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    visibility = pd.to_numeric(result.get(f"{prefix}visibility_m"), errors="coerce")
    wind = pd.to_numeric(result.get(f"{prefix}wind_speed_mps"), errors="coerce")
    gust = pd.to_numeric(result.get(f"{prefix}wind_gust_mps"), errors="coerce")
    ceiling = pd.to_numeric(result.get(f"{prefix}ceiling_m"), errors="coerce")
    precipitation = pd.to_numeric(result.get(f"{prefix}precipitation_mm"), errors="coerce")
    temperature = pd.to_numeric(result.get(f"{prefix}temperature_c"), errors="coerce")
    dew_point = pd.to_numeric(result.get(f"{prefix}dew_point_c"), errors="coerce")

    result[f"{prefix}low_visibility"] = (visibility < 5000).astype("int8")
    result[f"{prefix}strong_wind"] = ((wind >= 10.3) | (gust >= 15.4)).astype("int8")
    result[f"{prefix}low_ceiling"] = (ceiling < 300).astype("int8")
    result[f"{prefix}active_precipitation"] = (precipitation > 0).astype("int8")
    result[f"{prefix}freezing_conditions"] = (
        (temperature <= 1.0) & ((precipitation > 0) | ((temperature - dew_point).abs() <= 2.0))
    ).astype("int8")
    severe_flags = [
        result[f"{prefix}low_visibility"],
        result[f"{prefix}strong_wind"],
        result[f"{prefix}low_ceiling"],
        result[f"{prefix}freezing_conditions"],
        pd.to_numeric(result.get(f"{prefix}thunderstorm_flag", 0), errors="coerce").fillna(0),
        pd.to_numeric(result.get(f"{prefix}snow_flag", 0), errors="coerce").fillna(0),
    ]
    result[f"{prefix}weather_severity"] = sum(severe_flags).clip(0, 6).astype("int8")
    return result
