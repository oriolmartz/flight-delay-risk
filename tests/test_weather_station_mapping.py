import pandas as pd
import pytest

from src.weather.airport_station_map import load_airport_station_map


def test_default_mapping_is_unique_and_timezone_aware() -> None:
    mapping = load_airport_station_map()
    assert {"JFK", "LAX", "ORD", "ATL"}.issubset(set(mapping["iata"]))
    assert mapping["iata"].is_unique
    assert mapping["timezone"].str.contains("/").all()


def test_mapping_rejects_duplicate_iata(tmp_path) -> None:
    path = tmp_path / "mapping.csv"
    pd.DataFrame(
        [
            {"iata": "AAA", "icao": "KAAA", "station_id": "1", "timezone": "UTC", "latitude": 0, "longitude": 0},
            {"iata": "AAA", "icao": "KAAA", "station_id": "2", "timezone": "UTC", "latitude": 0, "longitude": 0},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate IATA"):
        load_airport_station_map(path)
