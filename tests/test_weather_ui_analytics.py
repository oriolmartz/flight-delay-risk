from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.weather.ui_analytics import (
    build_weather_ui_summary,
    load_weather_ui_summary,
    point_in_time_weather_snapshot,
    save_weather_ui_summary,
)


def _summary_frame() -> pd.DataFrame:
    rows = []
    for airport, destination in (("JFK", "LAX"), ("LAX", "JFK")):
        for hour in (8, 18):
            for adverse in (0, 1):
                for idx in range(6):
                    delayed = 1 if adverse and idx < 4 else int((not adverse) and idx == 0)
                    rows.append(
                        {
                            "Origin": airport,
                            "Dest": destination,
                            "CRSDepTime": hour * 100,
                            "CRSArrTime": ((hour + 3) % 24) * 100,
                            "ArrDel15": delayed,
                            "origin_weather_available": 1,
                            "destination_weather_available": 1,
                            "origin_weather_severity": adverse,
                            "destination_weather_severity": adverse,
                        }
                    )
    return pd.DataFrame(rows)


def test_build_weather_ui_summary_contains_map_and_hourly_layers(tmp_path: Path):
    payload = build_weather_ui_summary(
        _summary_frame(), top_airports=2, min_group_support=2
    )
    assert payload["schema_version"] == 1
    assert len(payload["airport_layers"]) == 2
    assert len(payload["hourly_heatmap"]) > 0
    jfk = next(row for row in payload["airport_layers"] if row["airport"] == "JFK")
    assert jfk["origin_weather_uplift"] > 0
    assert jfk["origin_weather_severity"] == 0.5

    path = save_weather_ui_summary(payload, tmp_path / "weather_ui_summary.json")
    loaded = load_weather_ui_summary(path)
    assert loaded["rows"] == len(_summary_frame())


def test_point_in_time_snapshot_never_uses_future_weather():
    flight = pd.DataFrame(
        [
            {
                "FlightDate": "2024-07-15",
                "CRSDepTime": 1800,
                "Origin": "JFK",
                "Dest": "LAX",
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {"iata": "JFK", "station_id": "JFK1"},
            {"iata": "LAX", "station_id": "LAX1"},
        ]
    )
    timezone_mapping = pd.DataFrame(
        [
            {"iata": "JFK", "timezone": "America/New_York"},
            {"iata": "LAX", "timezone": "America/Los_Angeles"},
        ]
    )
    weather = pd.DataFrame(
        [
            {
                "station_id": station,
                "observation_time_utc": timestamp,
                "temperature_c": 20.0,
                "dew_point_c": 10.0,
                "wind_speed_mps": 12.0 if station == "JFK1" else 3.0,
                "wind_gust_mps": 16.0 if station == "JFK1" else 4.0,
                "visibility_m": 9000.0,
                "ceiling_m": 1000.0,
                "precipitation_mm": 0.0,
                "sea_level_pressure_hpa": 1012.0,
                "fog_flag": 0,
                "thunderstorm_flag": 0,
                "snow_flag": 0,
                "quality_flag": "test",
            }
            for station in ("JFK1", "LAX1")
            for timestamp in (
                pd.Timestamp("2024-07-15T15:00:00Z"),
                pd.Timestamp("2024-07-15T17:00:00Z"),
            )
        ]
    )
    snapshot = point_in_time_weather_snapshot(
        flight,
        weather,
        mapping,
        timezone_mapping,
        prediction_horizon_hours=2,
        max_observation_age_hours=6,
    )
    # 18:00 New York = 22:00 UTC; cutoff = 20:00 UTC, so 17:00 is eligible.
    assert snapshot["available"] is True
    assert snapshot["origin"]["observation_time_utc"].startswith("2024-07-15T17:00:00")
    assert snapshot["origin"]["flags"]["strong_wind"] is True
