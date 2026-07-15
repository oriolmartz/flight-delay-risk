"""Load and normalize BTS Reporting Carrier On-Time Performance CSV files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

COLUMN_ALIASES: dict[str, str] = {
    "flightdate": "FlightDate",
    "fldate": "FlightDate",
    "year": "Year",
    "month": "Month",
    "dayofmonth": "DayOfMonth",
    "dayofweek": "DayOfWeek",
    "reportingairline": "Airline",
    "operatingairline": "Airline",
    "iatacodereportingairline": "Airline",
    "uniquecarrier": "Airline",
    "opuniquecarrier": "Airline",
    "opcarrier": "Airline",
    "carrier": "Airline",
    "origin": "Origin",
    "dest": "Dest",
    "destination": "Dest",
    "crsdeptime": "CRSDepTime",
    "crsarrtime": "CRSArrTime",
    "crselapsedtime": "CRSElapsedTime",
    "distance": "Distance",
    "arrdel15": "ArrDel15",
    "cancelled": "Cancelled",
    "diverted": "Diverted",
    "depdelay": "DepDelay",
    "depdelayminutes": "DepDelayMinutes",
    "arrdelay": "ArrDelay",
    "arrdelayminutes": "ArrDelayMinutes",
    "actualelapsedtime": "ActualElapsedTime",
    "airtime": "AirTime",
    "taxiout": "TaxiOut",
    "taxiin": "TaxiIn",
    "wheelsoff": "WheelsOff",
    "wheelson": "WheelsOn",
    "deptime": "DepTime",
    "arrtime": "ArrTime",
    "carrierdelay": "CarrierDelay",
    "weatherdelay": "WeatherDelay",
    "nasdelay": "NASDelay",
    "securitydelay": "SecurityDelay",
    "lateaircraftdelay": "LateAircraftDelay",
    "cancellationcode": "CancellationCode",
}


def _normalize_column_name(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: COLUMN_ALIASES[key]
        for col in df.columns
        if (key := _normalize_column_name(col)) in COLUMN_ALIASES
    }
    frame = df.rename(columns=rename_map)
    unnamed_cols = [c for c in frame.columns if str(c).lower().startswith("unnamed")]
    return frame.drop(columns=unnamed_cols) if unnamed_cols else frame


def _uniform_sample_csv(
    path: Path,
    n_rows: int,
    *,
    chunksize: int = 100_000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Uniformly sample the complete CSV instead of taking its first rows.

    Random priorities are generated for every row while streaming the file. The
    ``n_rows`` smallest priorities form an exact reservoir sample, so sorted BTS
    exports retain coverage across the whole month.
    """
    if n_rows <= 0:
        raise ValueError("max_rows must be positive")
    rng = np.random.default_rng(random_seed)
    reservoir: pd.DataFrame | None = None
    source_offset = 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        chunk = chunk.copy()
        chunk["__sample_priority"] = rng.random(len(chunk))
        chunk["__source_row"] = np.arange(source_offset, source_offset + len(chunk))
        source_offset += len(chunk)
        reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > n_rows:
            reservoir = reservoir.nsmallest(n_rows, "__sample_priority", keep="first")
    if reservoir is None:
        return pd.DataFrame()
    return (
        reservoir.sort_values("__source_row", kind="stable")
        .drop(columns=["__sample_priority", "__source_row"])
        .reset_index(drop=True)
    )


def load_raw_csv(
    path: Path,
    max_rows: int | None = None,
    *,
    chunksize: int = 100_000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    logger.info("Loading raw CSV: %s", path)
    if max_rows is None:
        frame = pd.read_csv(path, low_memory=False)
    else:
        logger.info("Uniformly sampling %d rows across %s", max_rows, Path(path).name)
        frame = _uniform_sample_csv(
            Path(path), max_rows, chunksize=chunksize, random_seed=random_seed
        )
    frame = normalize_columns(frame)
    logger.info("Loaded %d rows, %d columns from %s", len(frame), frame.shape[1], Path(path).name)
    return frame


def load_raw_directory(
    input_dir: Path = RAW_DATA_DIR,
    max_rows_per_file: int | None = None,
    *,
    duplicate_month_policy: str | None = None,
) -> pd.DataFrame:
    """Load all raw CSVs, optionally enforcing one explicit source per month."""
    input_dir = Path(input_dir)
    if duplicate_month_policy is None:
        csv_paths = sorted(input_dir.glob("*.csv"))
    else:
        from src.data.manifest import resolve_monthly_sources

        csv_paths, _ = resolve_monthly_sources(
            input_dir, duplicate_month_policy=duplicate_month_policy
        )
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}. See docs/DATA.md or run the local demo."
        )
    frames = [load_raw_csv(path, max_rows=max_rows_per_file) for path in csv_paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Combined %d files into %d rows", len(csv_paths), len(combined))
    return combined
