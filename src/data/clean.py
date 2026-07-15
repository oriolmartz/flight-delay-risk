"""Clean and validate raw BTS flight data before feature engineering.

The preparation layer is deliberately strict. FlightRisk is a temporal ML
system, so malformed dates, impossible schedule values, non-binary targets or
calendar inconsistencies are data-contract failures rather than values that
should silently flow into model training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import FORBIDDEN_LEAKAGE_COLUMNS, REQUIRED_RAW_COLUMNS, TARGET_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SchemaValidationError(ValueError):
    """Raised when the raw or cleaned frame violates the BTS data contract."""


@dataclass
class CleaningReport:
    input_rows: int
    missing_target_rows: int = 0
    cancelled_or_diverted_rows: int = 0
    invalid_date_rows: int = 0
    invalid_numeric_rows: int = 0
    invalid_schedule_rows: int = 0
    output_rows: int = 0
    exact_duplicate_rows_observed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns after normalization: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )


def _valid_hhmm(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    whole = numeric.notna() & np.isclose(numeric, np.floor(numeric))
    numeric = numeric.fillna(-1).astype(int)
    return whole & ((numeric == 2400) | ((numeric >= 0) & (numeric <= 2359) & (numeric % 100 < 60)))


def clean_flights_with_report(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a cleaned frame and auditable row-level cleaning counts."""
    validate_required_columns(df)
    frame = df.copy()
    report = CleaningReport(input_rows=len(frame))

    # Cancelled/diverted rows are classified before target validation so the
    # manifest reports why they were removed instead of counting them as
    # generic missing-target observations.
    cancelled_or_diverted = pd.Series(False, index=frame.index)
    if "Cancelled" in frame.columns:
        cancelled_or_diverted |= (
            pd.to_numeric(frame["Cancelled"], errors="coerce").fillna(0).ne(0)
        )
    if "Diverted" in frame.columns:
        cancelled_or_diverted |= (
            pd.to_numeric(frame["Diverted"], errors="coerce").fillna(0).ne(0)
        )
    report.cancelled_or_diverted_rows = int(cancelled_or_diverted.sum())
    frame = frame.loc[~cancelled_or_diverted].copy()

    target_numeric = pd.to_numeric(frame[TARGET_COL], errors="coerce")
    missing_target = target_numeric.isna()
    report.missing_target_rows = int(missing_target.sum())
    frame = frame.loc[~missing_target].copy()
    target_numeric = target_numeric.loc[~missing_target]

    invalid_target_values = ~target_numeric.isin([0, 1])
    if invalid_target_values.any():
        bad_values = sorted(target_numeric.loc[invalid_target_values].unique().tolist())[:10]
        raise SchemaValidationError(
            f"{TARGET_COL} must be binary 0/1 after normalization; found {bad_values}"
        )
    frame[TARGET_COL] = target_numeric.astype("int8")

    # Parse the temporal key during ingestion, never later in the modelling code.
    parsed_dates = pd.to_datetime(frame["FlightDate"], errors="coerce", format="mixed")
    invalid_dates = parsed_dates.isna()
    report.invalid_date_rows = int(invalid_dates.sum())
    frame = frame.loc[~invalid_dates].copy()
    frame["FlightDate"] = parsed_dates.loc[~invalid_dates].dt.normalize()

    numeric_cols = [
        "Year",
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
    ]
    numeric = {col: pd.to_numeric(frame[col], errors="coerce") for col in numeric_cols}
    invalid_numeric = pd.Series(False, index=frame.index)
    for values in numeric.values():
        invalid_numeric |= values.isna()
    report.invalid_numeric_rows = int(invalid_numeric.sum())
    frame = frame.loc[~invalid_numeric].copy()
    for col, values in numeric.items():
        frame[col] = values.loc[~invalid_numeric]

    schedule_valid = (
        frame["Year"].between(1987, 2100)
        & frame["Month"].between(1, 12)
        & frame["DayOfWeek"].between(1, 7)
        & _valid_hhmm(frame["CRSDepTime"])
        & _valid_hhmm(frame["CRSArrTime"])
        & frame["CRSElapsedTime"].gt(0)
        & frame["Distance"].gt(0)
    )
    report.invalid_schedule_rows = int((~schedule_valid).sum())
    frame = frame.loc[schedule_valid].copy()

    frame["Year"] = frame["Year"].astype("int16")
    frame["Month"] = frame["Month"].astype("int8")
    frame["DayOfWeek"] = frame["DayOfWeek"].astype("int8")
    frame["CRSDepTime"] = frame["CRSDepTime"].astype("int16")
    frame["CRSArrTime"] = frame["CRSArrTime"].astype("int16")
    frame["CRSElapsedTime"] = frame["CRSElapsedTime"].astype("float32")
    frame["Distance"] = frame["Distance"].astype("float32")

    # Calendar fields must agree with FlightDate. A mismatch indicates the wrong
    # month/file was combined or a malformed source export.
    year_mismatch = frame["Year"].ne(frame["FlightDate"].dt.year)
    month_mismatch = frame["Month"].ne(frame["FlightDate"].dt.month)
    weekday_mismatch = frame["DayOfWeek"].ne(frame["FlightDate"].dt.dayofweek + 1)
    if year_mismatch.any() or month_mismatch.any() or weekday_mismatch.any():
        raise SchemaValidationError(
            "Calendar columns are inconsistent with FlightDate: "
            f"year={int(year_mismatch.sum())}, month={int(month_mismatch.sum())}, "
            f"day_of_week={int(weekday_mismatch.sum())}."
        )

    for col in ["Origin", "Dest", "Airline"]:
        frame[col] = frame[col].astype("string").str.strip().str.upper()
        if frame[col].isna().any() or frame[col].eq("").any():
            raise SchemaValidationError(f"Column {col} contains missing or empty identifiers")

    cols_to_drop = [c for c in FORBIDDEN_LEAKAGE_COLUMNS if c in frame.columns]
    frame = frame.drop(columns=cols_to_drop)

    report.exact_duplicate_rows_observed = int(frame.duplicated(keep=False).sum())
    report.output_rows = len(frame)
    frame = frame.sort_values(["FlightDate", "CRSDepTime"], kind="stable").reset_index(drop=True)

    logger.info(
        "Cleaned %d -> %d rows (missing target=%d, cancelled/diverted=%d, "
        "invalid date=%d, invalid numeric=%d, invalid schedule=%d)",
        report.input_rows,
        report.output_rows,
        report.missing_target_rows,
        report.cancelled_or_diverted_rows,
        report.invalid_date_rows,
        report.invalid_numeric_rows,
        report.invalid_schedule_rows,
    )
    return frame, report


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible convenience wrapper returning only the clean frame."""
    cleaned, _ = clean_flights_with_report(df)
    return cleaned
