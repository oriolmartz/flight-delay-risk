"""Compact weather analytics used by the public dashboard.

The module keeps UI concerns separate from model training:
- point-in-time weather snapshots are resolved from the canonical NOAA table;
- airport and airport-hour summaries are built offline from the joined BTS frame;
- the reported weather uplift is descriptive association, never causal attribution.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.weather.model_features import WEATHER_MODEL_FEATURES
from src.weather.point_in_time_join import PointInTimeJoinConfig, attach_weather_context

SUMMARY_SCHEMA_VERSION = 1
DEFAULT_MIN_GROUP_SUPPORT = 100


def _weighted_rate(frame: pd.DataFrame, target: str = "ArrDel15") -> float | None:
    values = pd.to_numeric(frame[target], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _weather_uplift(
    frame: pd.DataFrame,
    *,
    severity_column: str,
    target: str = "ArrDel15",
    min_support: int = DEFAULT_MIN_GROUP_SUPPORT,
) -> dict[str, Any]:
    severity = pd.to_numeric(frame[severity_column], errors="coerce")
    target_values = pd.to_numeric(frame[target], errors="coerce")
    valid = severity.notna() & target_values.isin([0, 1])
    clear = target_values.loc[valid & severity.eq(0)]
    adverse = target_values.loc[valid & severity.gt(0)]
    clear_rate = float(clear.mean()) if len(clear) else None
    adverse_rate = float(adverse.mean()) if len(adverse) else None
    uplift = None
    if len(clear) >= min_support and len(adverse) >= min_support:
        uplift = float(adverse_rate - clear_rate)
    return {
        "clear_delay_rate": clear_rate,
        "adverse_delay_rate": adverse_rate,
        "weather_uplift": uplift,
        "clear_support": int(len(clear)),
        "adverse_support": int(len(adverse)),
    }


def _endpoint_summary(
    frame: pd.DataFrame,
    *,
    airport_column: str,
    prefix: str,
    min_support: int,
) -> pd.DataFrame:
    severity_column = f"{prefix}weather_severity"
    available_column = f"{prefix}weather_available"
    rows: list[dict[str, Any]] = []
    for airport, group in frame.groupby(airport_column, sort=True):
        available = pd.to_numeric(group.get(available_column, 0), errors="coerce").fillna(0).eq(1)
        observed = group.loc[available]
        severity = pd.to_numeric(observed.get(severity_column), errors="coerce")
        uplift = _weather_uplift(
            observed,
            severity_column=severity_column,
            min_support=min_support,
        )
        rows.append(
            {
                "airport": str(airport),
                "delay_rate": _weighted_rate(group),
                "support": int(len(group)),
                "weather_support": int(len(observed)),
                "weather_coverage": float(len(observed) / len(group)) if len(group) else 0.0,
                "weather_severity": _safe_mean(severity),
                "adverse_share": float(severity.gt(0).mean()) if len(severity.dropna()) else None,
                **uplift,
            }
        )
    return pd.DataFrame(rows)


def _merge_endpoint_summaries(origin: pd.DataFrame, destination: pd.DataFrame) -> list[dict[str, Any]]:
    origin = origin.add_prefix("origin_").rename(columns={"origin_airport": "airport"})
    destination = destination.add_prefix("destination_").rename(
        columns={"destination_airport": "airport"}
    )
    merged = origin.merge(destination, on="airport", how="outer")
    merged = merged.sort_values("airport", kind="stable")
    return merged.replace({np.nan: None}).to_dict(orient="records")


def _hourly_summary(
    frame: pd.DataFrame,
    *,
    airport_column: str,
    time_column: str,
    prefix: str,
    perspective: str,
    selected_airports: set[str],
    min_support: int,
) -> list[dict[str, Any]]:
    work = frame.loc[frame[airport_column].astype(str).isin(selected_airports)].copy()
    raw_time = work[time_column].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
    work["hour"] = pd.to_numeric(raw_time.str[:-2], errors="coerce").fillna(0).astype(int).mod(24)
    rows: list[dict[str, Any]] = []
    severity_column = f"{prefix}weather_severity"
    available_column = f"{prefix}weather_available"
    for (airport, hour), group in work.groupby([airport_column, "hour"], sort=True):
        available = pd.to_numeric(group.get(available_column, 0), errors="coerce").fillna(0).eq(1)
        observed = group.loc[available]
        severity = pd.to_numeric(observed.get(severity_column), errors="coerce")
        uplift = _weather_uplift(
            observed,
            severity_column=severity_column,
            min_support=min_support,
        )
        rows.append(
            {
                "airport": str(airport),
                "perspective": perspective,
                "hour": int(hour),
                "support": int(len(group)),
                "weather_support": int(len(observed)),
                "delay_rate": _weighted_rate(group),
                "weather_severity": _safe_mean(severity),
                "adverse_share": float(severity.gt(0).mean()) if len(severity.dropna()) else None,
                **uplift,
            }
        )
    return rows


def build_weather_ui_summary(
    frame: pd.DataFrame,
    *,
    top_airports: int = 30,
    min_group_support: int = DEFAULT_MIN_GROUP_SUPPORT,
) -> dict[str, Any]:
    """Build a compact dashboard artifact from the leakage-safe joined flight frame."""
    required = {
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "ArrDel15",
        "origin_weather_available",
        "destination_weather_available",
        "origin_weather_severity",
        "destination_weather_severity",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Weather UI summary requires columns: {sorted(missing)}")

    work = frame.loc[pd.to_numeric(frame["ArrDel15"], errors="coerce").isin([0, 1])].copy()
    origin = _endpoint_summary(
        work,
        airport_column="Origin",
        prefix="origin_",
        min_support=min_group_support,
    )
    destination = _endpoint_summary(
        work,
        airport_column="Dest",
        prefix="destination_",
        min_support=min_group_support,
    )
    combined_support = (
        origin.set_index("airport")["support"].add(
            destination.set_index("airport")["support"], fill_value=0
        )
    )
    selected_airports = set(
        combined_support.sort_values(ascending=False).head(top_airports).index.astype(str)
    )
    hourly = _hourly_summary(
        work,
        airport_column="Origin",
        time_column="CRSDepTime",
        prefix="origin_",
        perspective="origin",
        selected_airports=selected_airports,
        min_support=min_group_support,
    ) + _hourly_summary(
        work,
        airport_column="Dest",
        time_column="CRSArrTime",
        prefix="destination_",
        perspective="destination",
        selected_airports=selected_airports,
        min_support=min_group_support,
    )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "BTS 2024 joined to point-in-time NOAA observations",
        "target": "ArrDel15",
        "weather_definition": "adverse means weather_severity > 0",
        "uplift_definition": (
            "descriptive delay-rate difference between adverse and clear observations; "
            "association, not causal attribution"
        ),
        "rows": int(len(work)),
        "airport_layers": _merge_endpoint_summaries(origin, destination),
        "hourly_heatmap": hourly,
    }


def save_weather_ui_summary(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_weather_ui_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if int(payload.get("schema_version", 0)) != SUMMARY_SCHEMA_VERSION:
        return {}
    return payload


def point_in_time_weather_snapshot(
    raw_flight: pd.DataFrame,
    weather: pd.DataFrame,
    station_mapping: pd.DataFrame,
    timezone_mapping: pd.DataFrame,
    *,
    prediction_horizon_hours: int = 6,
    max_observation_age_hours: int = 6,
) -> dict[str, Any]:
    """Resolve one pre-departure weather snapshot using the production as-of join."""
    joined = attach_weather_context(
        raw_flight,
        weather,
        station_mapping,
        PointInTimeJoinConfig(
            prediction_horizon_hours=prediction_horizon_hours,
            max_observation_age_hours=max_observation_age_hours,
        ),
        timezone_mapping=timezone_mapping,
    )
    row = joined.iloc[0]
    endpoints: dict[str, dict[str, Any]] = {}
    feature_values: dict[str, Any] = {}
    for endpoint, prefix in (("origin", "origin_"), ("destination", "destination_")):
        available = bool(int(row.get(f"{prefix}weather_available", 0) or 0))
        endpoint_payload = {
            "available": available,
            "stale": bool(int(row.get(f"{prefix}weather_stale", 1) or 0)),
            "observation_time_utc": (
                pd.Timestamp(row[f"{prefix}observation_time_utc"]).isoformat()
                if available and pd.notna(row.get(f"{prefix}observation_time_utc"))
                else None
            ),
            "observation_age_minutes": (
                float(row[f"{prefix}observation_age_minutes"])
                if available and pd.notna(row.get(f"{prefix}observation_age_minutes"))
                else None
            ),
            "temperature_c": _python_value(row.get(f"{prefix}temperature_c")),
            "wind_speed_mps": _python_value(row.get(f"{prefix}wind_speed_mps")),
            "wind_gust_mps": _python_value(row.get(f"{prefix}wind_gust_mps")),
            "visibility_m": _python_value(row.get(f"{prefix}visibility_m")),
            "ceiling_m": _python_value(row.get(f"{prefix}ceiling_m")),
            "precipitation_mm": _python_value(row.get(f"{prefix}precipitation_mm")),
            "weather_severity": _python_value(row.get(f"{prefix}weather_severity")),
            "flags": {
                "low_visibility": _bool_value(row.get(f"{prefix}low_visibility", 0)),
                "strong_wind": _bool_value(row.get(f"{prefix}strong_wind", 0)),
                "low_ceiling": _bool_value(row.get(f"{prefix}low_ceiling", 0)),
                "active_precipitation": _bool_value(row.get(f"{prefix}active_precipitation", 0)),
                "freezing_conditions": _bool_value(row.get(f"{prefix}freezing_conditions", 0)),
                "thunderstorm": _bool_value(row.get(f"{prefix}thunderstorm_flag", 0)),
                "snow": _bool_value(row.get(f"{prefix}snow_flag", 0)),
                "fog": _bool_value(row.get(f"{prefix}fog_flag", 0)),
            },
        }
        endpoints[endpoint] = endpoint_payload
        for column in WEATHER_MODEL_FEATURES:
            if column.startswith(prefix):
                feature_values[column] = _python_value(row.get(column))
    both_available = bool(endpoints["origin"].get("available") and endpoints["destination"].get("available"))
    return {
        "available": both_available,
        "prediction_cutoff_utc": (
            pd.Timestamp(row["prediction_cutoff_utc"]).isoformat()
            if pd.notna(row.get("prediction_cutoff_utc"))
            else None
        ),
        "origin": endpoints["origin"],
        "destination": endpoints["destination"],
        "feature_values": feature_values,
    }


def _bool_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
