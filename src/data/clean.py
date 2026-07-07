"""
Clean and validate raw BTS flight data before feature engineering.

Responsibilities:
    * Validate that required columns are present.
    * Drop rows with a missing target.
    * Filter out cancelled / diverted flights (these never have a
      meaningful arrival outcome), WITHOUT keeping Cancelled/Diverted as
      model features (that filtering happens here, not in the feature
      builder, precisely so those columns never reach the model).
    * Coerce dtypes for the columns we rely on.
"""
from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_RAW_COLUMNS, TARGET_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SchemaValidationError(ValueError):
    """Raised when required columns are missing from the input data."""


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required columns after normalization: {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``df`` ready for feature engineering."""
    validate_required_columns(df)
    df = df.copy()

    n_start = len(df)

    # Drop rows with a missing target -- we cannot train or evaluate on them.
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    logger.info("Dropped %d rows with missing target", n_start - len(df))

    # Filter out cancelled / diverted flights if those columns are present.
    # These columns are used ONLY to filter rows here; they are removed
    # from the frame immediately after and are never used as features.
    n_before_filter = len(df)
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"].fillna(0).astype(float) == 0]
    if "Diverted" in df.columns:
        df = df[df["Diverted"].fillna(0).astype(float) == 0]
    logger.info(
        "Filtered %d cancelled/diverted rows", n_before_filter - len(df)
    )

    # Drop any leakage / post-flight columns now so they can never
    # accidentally flow into feature engineering.
    from src.config import FORBIDDEN_LEAKAGE_COLUMNS

    cols_to_drop = [c for c in FORBIDDEN_LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped forbidden leakage columns: %s", cols_to_drop)

    # Coerce core numeric columns.
    numeric_cols = ["Year", "Month", "DayOfWeek", "CRSDepTime", "CRSArrTime",
                     "CRSElapsedTime", "Distance"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])

    # String columns.
    for col in ["Origin", "Dest", "Airline"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    df = df.reset_index(drop=True)
    logger.info("Cleaned dataset has %d rows", len(df))
    return df
