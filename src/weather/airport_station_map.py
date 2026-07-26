"""Versioned airport-to-weather-station mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_MAPPING_PATH = Path("data/weather/airport_station_map.csv")
REQUIRED_COLUMNS = {
    "iata",
    "icao",
    "station_id",
    "timezone",
    "latitude",
    "longitude",
}


@dataclass(frozen=True)
class AirportStation:
    iata: str
    icao: str
    station_id: str
    timezone: str
    latitude: float
    longitude: float


def load_airport_station_map(path: str | Path = DEFAULT_MAPPING_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"iata": "string", "icao": "string", "station_id": "string"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Airport-station mapping missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ("iata", "icao", "station_id", "timezone"):
        frame[column] = frame[column].astype("string").str.strip()
    frame["iata"] = frame["iata"].str.upper()
    frame["icao"] = frame["icao"].str.upper()
    if frame["iata"].duplicated().any():
        duplicates = frame.loc[frame["iata"].duplicated(keep=False), "iata"].tolist()
        raise ValueError(f"Duplicate IATA mappings: {sorted(set(duplicates))}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Airport-station mapping contains null required values")
    return frame.sort_values("iata", kind="stable").reset_index(drop=True)
