"""Normalize NOAA-style weather extracts into a canonical hourly schema."""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from .schemas import BINARY_WEATHER_COLUMNS, NUMERIC_WEATHER_COLUMNS, WEATHER_COLUMNS

_ALIASES = {
    "STATION": "station_id",
    "station": "station_id",
    "DATE": "observation_time_utc",
    "date": "observation_time_utc",
    "TMP": "temperature_c",
    "DEW": "dew_point_c",
    "WND": "wind_speed_mps",
    "VIS": "visibility_m",
    "SLP": "sea_level_pressure_hpa",
}

_MISSING_TOKENS = {"", "+9999", "9999", "99999", "999999", "+99999"}


def _parse_scaled_value(value: object, *, scale: float = 1.0) -> float | None:
    if pd.isna(value):
        return None
    token = str(value).split(",", 1)[0].strip()
    if token in _MISSING_TOKENS:
        return None
    try:
        return float(token) / scale
    except ValueError:
        return None


def _parse_wind(value: object) -> float | None:
    if pd.isna(value):
        return None
    parts = str(value).split(",")
    if len(parts) < 4 or parts[3].strip() in _MISSING_TOKENS:
        return None
    try:
        return float(parts[3]) / 10.0
    except ValueError:
        return None


def _parse_visibility(value: object) -> float | None:
    return _parse_scaled_value(value, scale=1.0)


def _parse_ceiling(value: object) -> float | None:
    """Parse NOAA CIG ceiling height.

    CIG stores the ceiling height in metres as its first component. 99999 is
    the missing/unlimited sentinel and remains null in the canonical dataset.
    """
    return _parse_scaled_value(value, scale=1.0)


def _parse_precipitation_group(value: object) -> tuple[int, float] | None:
    """Parse one NOAA AA* liquid-precipitation group.

    The first component is the accumulation period in hours and the second is
    liquid depth in tenths of a millimetre. Invalid/sentinel values are ignored.
    """
    if pd.isna(value):
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) < 2:
        return None
    period_token, depth_token = parts[0], parts[1]
    if period_token in _MISSING_TOKENS or depth_token in _MISSING_TOKENS:
        return None
    try:
        period_hours = int(period_token)
        depth_mm = float(depth_token) / 10.0
    except ValueError:
        return None
    if period_hours <= 0 or depth_mm < 0:
        return None
    return period_hours, depth_mm


def _select_precipitation(values: Iterable[object]) -> float | None:
    """Choose the shortest valid AA* accumulation period deterministically."""
    candidates = [parsed for value in values if (parsed := _parse_precipitation_group(value))]
    if not candidates:
        return None
    period_hours, depth_mm = min(candidates, key=lambda item: (item[0], item[1]))
    del period_hours
    return depth_mm


def _parse_gust(value: object) -> float | None:
    """Parse an optional NOAA OC1/GUST wind-gust speed in tenths of m/s."""
    return _parse_scaled_value(value, scale=10.0)


def _present_weather_flags(row: pd.Series) -> tuple[int, int, int]:
    text = " ".join(
        str(row.get(col, ""))
        for col in row.index
        if re.fullmatch(r"(AA|AW|MW)\d+", str(col))
    )
    upper = text.upper()
    return (
        int(any(code in upper for code in ("FG", "FZFG"))),
        int(any(code in upper for code in ("TS", "TSRA", "VCTS"))),
        int(any(code in upper for code in ("SN", "SG", "BLSN"))),
    )


def normalize_weather_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a typed canonical weather frame.

    Supports canonical columns directly and common NOAA Global Hourly CSV fields.
    Unknown optional fields remain null rather than being imputed during ingestion.
    """
    frame = raw.rename(columns={k: v for k, v in _ALIASES.items() if k in raw.columns}).copy()
    if "station_id" not in frame or "observation_time_utc" not in frame:
        raise ValueError("Weather data requires station and observation timestamp columns")

    if "temperature_c" in raw.columns:
        frame["temperature_c"] = pd.to_numeric(raw["temperature_c"], errors="coerce")
    elif "TMP" in raw.columns:
        frame["temperature_c"] = raw["TMP"].map(lambda x: _parse_scaled_value(x, scale=10.0))

    if "dew_point_c" in raw.columns:
        frame["dew_point_c"] = pd.to_numeric(raw["dew_point_c"], errors="coerce")
    elif "DEW" in raw.columns:
        frame["dew_point_c"] = raw["DEW"].map(lambda x: _parse_scaled_value(x, scale=10.0))

    if "wind_speed_mps" in raw.columns:
        frame["wind_speed_mps"] = pd.to_numeric(raw["wind_speed_mps"], errors="coerce")
    elif "WND" in raw.columns:
        frame["wind_speed_mps"] = raw["WND"].map(_parse_wind)

    if "wind_gust_mps" in raw.columns:
        frame["wind_gust_mps"] = pd.to_numeric(raw["wind_gust_mps"], errors="coerce")
    elif "OC1" in raw.columns:
        frame["wind_gust_mps"] = raw["OC1"].map(_parse_gust)
    elif "GUST" in raw.columns:
        frame["wind_gust_mps"] = raw["GUST"].map(_parse_gust)

    if "visibility_m" in raw.columns:
        frame["visibility_m"] = pd.to_numeric(raw["visibility_m"], errors="coerce")
    elif "VIS" in raw.columns:
        frame["visibility_m"] = raw["VIS"].map(_parse_visibility)

    if "ceiling_m" in raw.columns:
        frame["ceiling_m"] = pd.to_numeric(raw["ceiling_m"], errors="coerce")
    elif "CIG" in raw.columns:
        frame["ceiling_m"] = raw["CIG"].map(_parse_ceiling)

    if "precipitation_mm" in raw.columns:
        frame["precipitation_mm"] = pd.to_numeric(raw["precipitation_mm"], errors="coerce")
    else:
        precipitation_columns = sorted(
            (column for column in raw.columns if re.fullmatch(r"AA\d+", str(column))),
            key=lambda column: int(str(column)[2:]),
        )
        if precipitation_columns:
            frame["precipitation_mm"] = raw[precipitation_columns].apply(
                lambda row: _select_precipitation(row.tolist()), axis=1
            )

    if "sea_level_pressure_hpa" in raw.columns:
        frame["sea_level_pressure_hpa"] = pd.to_numeric(
            raw["sea_level_pressure_hpa"], errors="coerce"
        )
    elif "SLP" in raw.columns:
        frame["sea_level_pressure_hpa"] = raw["SLP"].map(
            lambda x: _parse_scaled_value(x, scale=10.0)
        )

    frame["station_id"] = frame["station_id"].astype("string").str.strip()
    frame["observation_time_utc"] = pd.to_datetime(
        frame["observation_time_utc"], utc=True, errors="coerce", format="mixed"
    )

    for column in NUMERIC_WEATHER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if not set(BINARY_WEATHER_COLUMNS).issubset(frame.columns):
        flags = frame.apply(_present_weather_flags, axis=1, result_type="expand")
        flags.columns = BINARY_WEATHER_COLUMNS
        for column in BINARY_WEATHER_COLUMNS:
            if column not in frame:
                frame[column] = flags[column]
    for column in BINARY_WEATHER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")

    if "quality_flag" not in frame:
        frame["quality_flag"] = "unknown"
    frame["quality_flag"] = frame["quality_flag"].astype("string").fillna("unknown")

    frame = frame.dropna(subset=["station_id", "observation_time_utc"])
    frame = frame[WEATHER_COLUMNS].sort_values(
        ["station_id", "observation_time_utc"], kind="stable"
    )
    return frame.drop_duplicates(
        ["station_id", "observation_time_utc"], keep="last"
    ).reset_index(drop=True)
