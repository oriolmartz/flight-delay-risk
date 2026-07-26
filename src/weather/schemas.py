"""Canonical schemas used by the weather ingestion layer."""
from __future__ import annotations

WEATHER_COLUMNS = [
    "station_id",
    "observation_time_utc",
    "temperature_c",
    "dew_point_c",
    "wind_speed_mps",
    "wind_gust_mps",
    "visibility_m",
    "ceiling_m",
    "precipitation_mm",
    "sea_level_pressure_hpa",
    "fog_flag",
    "thunderstorm_flag",
    "snow_flag",
    "quality_flag",
]

NUMERIC_WEATHER_COLUMNS = [
    "temperature_c",
    "dew_point_c",
    "wind_speed_mps",
    "wind_gust_mps",
    "visibility_m",
    "ceiling_m",
    "precipitation_mm",
    "sea_level_pressure_hpa",
]

BINARY_WEATHER_COLUMNS = ["fog_flag", "thunderstorm_flag", "snow_flag"]
