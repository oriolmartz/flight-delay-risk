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
    """Split chronologically: earliest (1 - test_size) rows train, rest test."""
    df_sorted = df.sort_values("FlightDate").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:cutoff_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[cutoff_idx:].reset_index(drop=True)
    logger.info(
        "Time-aware split: train=%d rows (up to %s), test=%d rows (from %s)",
        len(train_df),
        train_df["FlightDate"].max() if len(train_df) else "n/a",
        len(test_df),
        test_df["FlightDate"].min() if len(test_df) else "n/a",
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
            df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
            if df["FlightDate"].notna().sum() > 0:
                return time_aware_split(df.dropna(subset=["FlightDate"]), test_size)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Time-aware split failed (%s); falling back to stratified split", exc)
    return stratified_split(df, test_size)
