"""Leakage-safe as-of joins between flights and weather observations."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_builder import build_weather_features
from .schemas import BINARY_WEATHER_COLUMNS, NUMERIC_WEATHER_COLUMNS


@dataclass(frozen=True)
class PointInTimeJoinConfig:
    prediction_horizon_hours: int = 6
    max_observation_age_hours: int = 6
    flight_date_column: str = "FlightDate"
    departure_time_column: str = "CRSDepTime"
    origin_column: str = "Origin"
    destination_column: str = "Dest"


def _scheduled_departure_utc(flights: pd.DataFrame, timezone_mapping: pd.DataFrame, config: PointInTimeJoinConfig) -> pd.Series:
    dates = pd.to_datetime(flights[config.flight_date_column], errors="coerce", format="mixed")
    raw_time = flights[config.departure_time_column].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(4)
    hours = pd.to_numeric(raw_time.str[:-2], errors="coerce").fillna(0).astype(int)
    minutes = pd.to_numeric(raw_time.str[-2:], errors="coerce").fillna(0).astype(int)
    rollover = hours.eq(24)
    hours = hours.mod(24)
    naive = dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(rollover.astype(int), unit="D")

    timezone_lookup = timezone_mapping.set_index("iata")["timezone"].to_dict()
    output = pd.Series(pd.NaT, index=flights.index, dtype="datetime64[ns, UTC]")
    timezone_series = flights[config.origin_column].map(timezone_lookup)
    for timezone in timezone_series.dropna().unique():
        mask = timezone_series.eq(timezone)
        localized = naive.loc[mask].dt.tz_localize(str(timezone), ambiguous="NaT", nonexistent="shift_forward")
        output.loc[mask] = localized.dt.tz_convert("UTC")
    return output


def _attach_side(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    airport_column: str,
    prefix: str,
    max_age_hours: int,
) -> pd.DataFrame:
    station_lookup = mapping.set_index("iata")["station_id"]
    left = flights.copy()
    left["_station_id"] = left[airport_column].map(station_lookup).astype("string")
    left["_row_order"] = range(len(left))
    right = weather.rename(columns={"station_id": "_station_id"}).copy()
    right["_station_id"] = right["_station_id"].astype("string")

    observation_columns = NUMERIC_WEATHER_COLUMNS + BINARY_WEATHER_COLUMNS + ["quality_flag"]
    renamed = {column: f"{prefix}{column}" for column in observation_columns}
    timestamp_column = f"{prefix}observation_time_utc"
    right = right[["_station_id", "observation_time_utc", *observation_columns]].rename(columns=renamed)
    right = right.rename(columns={"observation_time_utc": timestamp_column})

    valid = left["_station_id"].notna() & left["prediction_cutoff_utc"].notna()
    pieces: list[pd.DataFrame] = []
    if valid.any():
        # Perform the as-of join independently per weather station.  A single
        # merge_asof(..., by="_station_id") is sensitive to global ordering
        # when destination stations are interleaved across origin time zones;
        # on large route networks that can return an older observation for the
        # destination side and incorrectly mark otherwise valid weather stale.
        weather_by_station = {
            station_id: station_weather.drop(columns="_station_id").sort_values(timestamp_column)
            for station_id, station_weather in right.groupby("_station_id", sort=False)
        }
        for station_id, station_flights in left.loc[valid].groupby("_station_id", sort=False):
            station_weather = weather_by_station.get(station_id)
            if station_weather is None or station_weather.empty:
                unmatched_station = station_flights.copy()
                unmatched_station[timestamp_column] = pd.Series(
                    pd.NaT,
                    index=unmatched_station.index,
                    dtype="datetime64[ns, UTC]",
                )
                for column in observation_columns:
                    unmatched_station[f"{prefix}{column}"] = pd.NA
                pieces.append(unmatched_station)
                continue

            matched_station = pd.merge_asof(
                station_flights.sort_values("prediction_cutoff_utc"),
                station_weather,
                left_on="prediction_cutoff_utc",
                right_on=timestamp_column,
                direction="backward",
                allow_exact_matches=True,
            )
            matched_station["_station_id"] = station_id
            pieces.append(matched_station)
    if (~valid).any():
        unmatched = left.loc[~valid].copy()
        unmatched[timestamp_column] = pd.Series(pd.NaT, index=unmatched.index, dtype="datetime64[ns, UTC]")
        for column in observation_columns:
            unmatched[f"{prefix}{column}"] = pd.NA
        pieces.append(unmatched)

    joined = pd.concat(pieces, ignore_index=True, sort=False).sort_values("_row_order", kind="stable")
    joined = joined.drop(columns=["_row_order", "_station_id"]).reset_index(drop=True)
    joined[timestamp_column] = pd.to_datetime(joined[timestamp_column], utc=True, errors="coerce")
    joined["prediction_cutoff_utc"] = pd.to_datetime(joined["prediction_cutoff_utc"], utc=True, errors="coerce")

    age_minutes = (joined["prediction_cutoff_utc"] - joined[timestamp_column]).dt.total_seconds() / 60.0
    joined[f"{prefix}observation_age_minutes"] = age_minutes
    stale = age_minutes.gt(max_age_hours * 60) | age_minutes.isna()
    joined[f"{prefix}weather_available"] = (~stale).astype("int8")
    joined[f"{prefix}weather_stale"] = stale.astype("int8")
    weather_value_columns = [f"{prefix}{column}" for column in observation_columns]
    joined.loc[stale, weather_value_columns] = pd.NA
    joined.loc[stale, timestamp_column] = pd.NaT
    joined.loc[stale, f"{prefix}observation_age_minutes"] = pd.NA
    return build_weather_features(joined, prefix=prefix)


def attach_weather_context(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
    mapping: pd.DataFrame,
    config: PointInTimeJoinConfig | None = None,
    *,
    timezone_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach latest origin and destination observations known at prediction cutoff."""
    config = config or PointInTimeJoinConfig()
    required_flight = {
        config.flight_date_column,
        config.departure_time_column,
        config.origin_column,
        config.destination_column,
    }
    missing = required_flight.difference(flights.columns)
    if missing:
        raise ValueError(f"Flights missing columns required for weather join: {sorted(missing)}")
    required_weather = {"station_id", "observation_time_utc"}
    if not required_weather.issubset(weather.columns):
        raise ValueError("Weather frame is not canonical; normalize it before joining")

    result = flights.copy()
    timezone_mapping = mapping if timezone_mapping is None else timezone_mapping
    required_timezone = {"iata", "timezone"}
    if not required_timezone.issubset(timezone_mapping.columns):
        raise ValueError("Timezone mapping must contain iata and timezone columns")
    result["scheduled_departure_utc"] = _scheduled_departure_utc(result, timezone_mapping, config)
    result["prediction_cutoff_utc"] = result["scheduled_departure_utc"] - pd.to_timedelta(
        config.prediction_horizon_hours, unit="h"
    )
    result = _attach_side(
        result,
        weather,
        mapping,
        airport_column=config.origin_column,
        prefix="origin_",
        max_age_hours=config.max_observation_age_hours,
    )
    result = _attach_side(
        result,
        weather,
        mapping,
        airport_column=config.destination_column,
        prefix="destination_",
        max_age_hours=config.max_observation_age_hours,
    )

    for prefix in ("origin_", "destination_"):
        observed = result[f"{prefix}observation_time_utc"]
        invalid = observed.notna() & observed.gt(result["prediction_cutoff_utc"])
        if invalid.any():
            raise AssertionError(f"Point-in-time contract violated for {prefix.rstrip('_')}")
    return result
