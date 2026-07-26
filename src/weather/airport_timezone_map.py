"""Versioned airport-to-IANA-timezone mapping used for flight cutoffs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_TIMEZONE_MAPPING_PATH = Path("data/weather/airport_timezone_map.csv")
REQUIRED_COLUMNS = {"iata", "timezone"}


def load_airport_timezone_map(path: str | Path = DEFAULT_TIMEZONE_MAPPING_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"iata": "string", "timezone": "string"})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Airport-timezone mapping missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["iata"] = frame["iata"].astype("string").str.strip().str.upper()
    frame["timezone"] = frame["timezone"].astype("string").str.strip()
    if frame["iata"].duplicated().any():
        duplicates = frame.loc[frame["iata"].duplicated(keep=False), "iata"].tolist()
        raise ValueError(f"Duplicate IATA timezone mappings: {sorted(set(duplicates))}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Airport-timezone mapping contains null required values")
    return frame.sort_values("iata", kind="stable").reset_index(drop=True)
