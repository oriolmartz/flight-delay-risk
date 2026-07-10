"""
Train / test splitting.

If ``FlightDate`` is available we use a time-aware split: earlier months
train, later months test. This mirrors the real deployment scenario
(predicting future flights from historical patterns) and avoids the
optimistic bias of a random split on time-series-like data.

If ``FlightDate`` is not available (e.g. some BTS exports only include
Year/Month), we fall back to a stratified random split on the target.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, TARGET_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


def time_aware_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically without placing the same timestamp in both sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    df_sorted = df.copy()
    df_sorted["FlightDate"] = pd.to_datetime(df_sorted["FlightDate"], errors="raise", format="mixed")
    df_sorted = df_sorted.sort_values("FlightDate").reset_index(drop=True)
    unique_dates = pd.Index(df_sorted["FlightDate"].drop_duplicates())
    if len(unique_dates) < 2:
        raise ValueError("Time-aware split requires at least two distinct FlightDate values")

    cutoff_position = int(len(unique_dates) * (1 - test_size))
    cutoff_position = min(max(cutoff_position, 1), len(unique_dates) - 1)
    cutoff_date = unique_dates[cutoff_position]

    train_df = df_sorted[df_sorted["FlightDate"] < cutoff_date].reset_index(drop=True)
    test_df = df_sorted[df_sorted["FlightDate"] >= cutoff_date].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Time-aware split produced an empty train or test partition")
    if train_df["FlightDate"].max() >= test_df["FlightDate"].min():
        raise AssertionError("Temporal boundary overlap detected")

    logger.info(
        "Time-aware split: train=%d rows (up to %s), test=%d rows (from %s)",
        len(train_df),
        train_df["FlightDate"].max(),
        len(test_df),
        test_df["FlightDate"].min(),
    )
    return train_df, test_df


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fall back split used when there is no date column to sort on."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        "Stratified random split: train=%d rows, test=%d rows", len(train_df), len(test_df)
    )
    return train_df, test_df


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the appropriate split strategy based on available columns."""
    if "FlightDate" in df.columns and df["FlightDate"].notna().any():
        try:
            df = df.copy()
            df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce", format="mixed")
            if df["FlightDate"].notna().sum() > 0:
                return time_aware_split(df.dropna(subset=["FlightDate"]), test_size)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Time-aware split failed (%s); falling back to stratified split", exc)
    return stratified_split(df, test_size)
