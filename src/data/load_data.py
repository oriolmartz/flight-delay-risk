"""
Load raw BTS Reporting Carrier On-Time Performance CSV files.

BTS TranStats exports use inconsistent column naming across years
(e.g. ``Reporting_Airline`` vs ``Operating_Airline`` vs ``UniqueCarrier``,
or a trailing unnamed index column). This module normalizes column names
so the rest of the pipeline can rely on a single stable schema.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Maps many possible raw BTS column spellings -> our canonical column name.
# Keys are normalized by lowercasing and removing spaces/underscores/punctuation,
# so BTS names like ``CRS_DEP_TIME`` and portfolio names like ``CRSDepTime``
# resolve to the same canonical field.
COLUMN_ALIASES: dict[str, str] = {
    "flightdate": "FlightDate",
    "fldate": "FlightDate",
    "year": "Year",
    "month": "Month",
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
    """Normalize raw BTS/export column names for alias matching.

    BTS exports may use names such as ``CRS_DEP_TIME``, while the sample
    data and older exports may use ``CRSDepTime``. Removing all non
    alphanumeric characters makes both variants become ``crsdeptime``.
    """
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw BTS columns to our canonical schema.

    Unknown columns are left untouched (rather than dropped) so nothing
    is silently lost; unused columns are simply ignored downstream.
    """
    rename_map = {}
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)

    # BTS exports frequently include a trailing "Unnamed: 27"-style index column.
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    return df


def load_raw_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a single BTS CSV file and normalize its columns.

    Args:
        path: CSV path to load.
        max_rows: Optional row cap applied at read time. This is useful for
            fast local experiments with large monthly BTS files because pandas
            does not need to load the entire month before sampling.
    """
    logger.info("Loading raw CSV: %s", path)
    if max_rows is not None:
        logger.info("Applying read-time row cap: max %d rows from %s", max_rows, path.name)
    df = pd.read_csv(path, low_memory=False, nrows=max_rows)
    df = normalize_columns(df)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path.name)
    return df


def load_raw_directory(input_dir: Path = RAW_DATA_DIR, max_rows_per_file: int | None = None) -> pd.DataFrame:
    """Load and concatenate every CSV file found in ``input_dir``.

    Raises:
        FileNotFoundError: if no CSV files are found in the directory.
    """
    input_dir = Path(input_dir)
    csv_paths = sorted(input_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}. "
            "Download monthly Reporting Carrier On-Time Performance CSVs from "
            "BTS TranStats and place them in this folder. "
            "See docs/DATA.md for instructions, or run "
            "'python -m scripts.run_local_demo' to use the bundled sample data instead."
        )

    frames = [load_raw_csv(p, max_rows=max_rows_per_file) for p in csv_paths]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if max_rows_per_file is not None:
        logger.info(
            "Combined %d files into %d total rows with max_rows_per_file=%d",
            len(csv_paths),
            len(combined),
            max_rows_per_file,
        )
    else:
        logger.info("Combined %d files into %d total rows", len(csv_paths), len(combined))
    return combined
