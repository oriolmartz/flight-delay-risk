import pandas as pd

from src.weather.parser import normalize_weather_frame
from src.weather.point_in_time_join import PointInTimeJoinConfig, attach_weather_context


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"iata": "JFK", "icao": "KJFK", "station_id": "JFKWX", "timezone": "America/New_York", "latitude": 0.0, "longitude": 0.0},
            {"iata": "LAX", "icao": "KLAX", "station_id": "LAXWX", "timezone": "America/Los_Angeles", "latitude": 0.0, "longitude": 0.0},
        ]
    )


def _weather() -> pd.DataFrame:
    return normalize_weather_frame(
        pd.DataFrame(
            {
                "station_id": ["JFKWX", "JFKWX", "LAXWX"],
                "observation_time_utc": [
                    "2024-01-01T15:00:00Z",
                    "2024-01-01T17:00:00Z",
                    "2024-01-01T15:30:00Z",
                ],
                "temperature_c": [0.0, 99.0, 12.0],
                "visibility_m": [3000.0, 100.0, 10000.0],
                "wind_speed_mps": [12.0, 30.0, 2.0],
            }
        )
    )


def test_join_uses_only_observations_available_at_cutoff() -> None:
    flights = pd.DataFrame({"FlightDate": ["2024-01-01"], "CRSDepTime": [1600], "Origin": ["JFK"], "Dest": ["LAX"]})
    result = attach_weather_context(
        flights,
        _weather(),
        _mapping(),
        PointInTimeJoinConfig(prediction_horizon_hours=6, max_observation_age_hours=8),
    )
    # 16:00 New York = 21:00 UTC, so T-6 cutoff is 15:00 UTC.
    assert result.loc[0, "prediction_cutoff_utc"] == pd.Timestamp("2024-01-01T15:00:00Z")
    assert result.loc[0, "origin_observation_time_utc"] == pd.Timestamp("2024-01-01T15:00:00Z")
    assert result.loc[0, "origin_temperature_c"] == 0.0
    assert result.loc[0, "origin_temperature_c"] != 99.0
    assert result.loc[0, "origin_low_visibility"] == 1
    assert result.loc[0, "origin_strong_wind"] == 1


def test_stale_weather_is_removed_and_flagged() -> None:
    flights = pd.DataFrame({"FlightDate": ["2024-01-01"], "CRSDepTime": [2200], "Origin": ["JFK"], "Dest": ["LAX"]})
    result = attach_weather_context(
        flights,
        _weather(),
        _mapping(),
        PointInTimeJoinConfig(prediction_horizon_hours=1, max_observation_age_hours=1),
    )
    assert result.loc[0, "origin_weather_available"] == 0
    assert result.loc[0, "origin_weather_stale"] == 1
    assert pd.isna(result.loc[0, "origin_temperature_c"])


def test_unknown_airport_degrades_to_missing_weather() -> None:
    flights = pd.DataFrame({"FlightDate": ["2024-01-01"], "CRSDepTime": [1600], "Origin": ["ZZZ"], "Dest": ["LAX"]})
    result = attach_weather_context(flights, _weather(), _mapping())
    assert result.loc[0, "origin_weather_available"] == 0
    assert pd.isna(result.loc[0, "origin_observation_time_utc"])



def test_destination_join_selects_latest_observation_per_station() -> None:
    mapping = pd.DataFrame(
        [
            {"iata": "JFK", "icao": "KJFK", "station_id": "JFKWX", "timezone": "America/New_York", "latitude": 0.0, "longitude": 0.0},
            {"iata": "LAX", "icao": "KLAX", "station_id": "LAXWX", "timezone": "America/Los_Angeles", "latitude": 0.0, "longitude": 0.0},
        ]
    )
    flights = pd.DataFrame(
        {
            "FlightDate": ["2024-01-15", "2024-01-15", "2024-01-15", "2024-01-15"],
            "CRSDepTime": [1800, 1800, 2000, 2000],
            "Origin": ["JFK", "LAX", "JFK", "LAX"],
            "Dest": ["LAX", "JFK", "LAX", "JFK"],
        }
    )
    timestamps = [
        "2024-01-15T16:00:00Z",
        "2024-01-15T17:00:00Z",
        "2024-01-15T18:00:00Z",
        "2024-01-15T19:00:00Z",
        "2024-01-15T20:00:00Z",
        "2024-01-15T22:00:00Z",
    ]
    weather = normalize_weather_frame(
        pd.DataFrame(
            {
                "station_id": ["JFKWX"] * len(timestamps) + ["LAXWX"] * len(timestamps),
                "observation_time_utc": timestamps + timestamps,
                "temperature_c": list(range(len(timestamps))) + list(range(10, 10 + len(timestamps))),
                "visibility_m": [10000.0] * (2 * len(timestamps)),
                "wind_speed_mps": [2.0] * (2 * len(timestamps)),
            }
        )
    )

    result = attach_weather_context(
        flights,
        weather,
        mapping,
        PointInTimeJoinConfig(prediction_horizon_hours=6, max_observation_age_hours=6),
    )

    assert result["origin_weather_available"].tolist() == [1, 1, 1, 1]
    assert result["destination_weather_available"].tolist() == [1, 1, 1, 1]
    assert (result["origin_observation_time_utc"] <= result["prediction_cutoff_utc"]).all()
    assert (result["destination_observation_time_utc"] <= result["prediction_cutoff_utc"]).all()
    # Every side must get the latest observation at or before its own cutoff.
    assert result["destination_observation_age_minutes"].tolist() == [0.0, 0.0, 0.0, 0.0]
