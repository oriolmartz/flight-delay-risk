"""Strict chronological partitioning for model selection and evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.split import time_aware_split


@dataclass(frozen=True)
class TemporalPartitions:
    model_train: pd.DataFrame
    selection: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "model_train": self.model_train,
            "selection": self.selection,
            "calibration": self.calibration,
            "test": self.test,
        }


def _assert_order(partitions: dict[str, pd.DataFrame]) -> None:
    previous_end: pd.Timestamp | None = None
    for name, frame in partitions.items():
        if frame.empty:
            raise ValueError(f"Temporal partition {name} is empty")
        dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
        start, end = dates.min(), dates.max()
        if previous_end is not None and previous_end >= start:
            raise AssertionError(f"Temporal overlap before {name}: {previous_end} >= {start}")
        previous_end = end


def split_model_selection_calibration_test(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    calibration_size: float = 0.15,
    selection_size: float = 0.20,
) -> TemporalPartitions:
    """Create four non-overlapping chronological blocks.

    ``test_size`` is relative to the full dataset. ``calibration_size`` is
    relative to the pre-test development period, and ``selection_size`` is
    relative to the remaining pre-calibration period.
    """
    if "FlightDate" not in df.columns:
        raise KeyError("FlightDate is required for strict temporal evaluation")
    development, test = time_aware_split(df, test_size=test_size)
    pre_calibration, calibration = time_aware_split(
        development, test_size=calibration_size
    )
    model_train, selection = time_aware_split(
        pre_calibration, test_size=selection_size
    )
    result = TemporalPartitions(model_train, selection, calibration, test)
    _assert_order(result.as_dict())
    return result


def _split_index(index: pd.Index, n_splits: int) -> list[pd.Index]:
    n = len(index)
    base, remainder = divmod(n, n_splits)
    blocks: list[pd.Index] = []
    start = 0
    for block_id in range(n_splits):
        size = base + (1 if block_id < remainder else 0)
        blocks.append(index[start : start + size])
        start += size
    return blocks


def make_expanding_time_folds(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding train/future-validation folds at date granularity."""
    if "FlightDate" not in df.columns:
        raise KeyError("Expected FlightDate column for temporal backtesting.")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be between 0 and 1")

    ordered = df.copy()
    ordered["FlightDate"] = pd.to_datetime(
        ordered["FlightDate"], errors="coerce", format="mixed"
    )
    ordered = (
        ordered.dropna(subset=["FlightDate"])
        .sort_values("FlightDate", kind="stable")
        .reset_index(drop=True)
    )
    unique_dates = pd.Index(ordered["FlightDate"].drop_duplicates())
    if len(unique_dates) < n_splits + 2:
        raise ValueError("Not enough distinct dates for the requested temporal folds")

    initial_date_count = max(1, int(len(unique_dates) * min_train_fraction))
    remaining_dates = unique_dates[initial_date_count:]
    if len(remaining_dates) < n_splits:
        raise ValueError("Not enough future dates for the requested folds")

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for block in _split_index(remaining_dates, n_splits):
        if len(block) == 0:
            continue
        test_start, test_end = block[0], block[-1]
        train = ordered.loc[ordered["FlightDate"] < test_start].reset_index(drop=True)
        future = ordered.loc[
            ordered["FlightDate"].between(test_start, test_end, inclusive="both")
        ].reset_index(drop=True)
        if train.empty or future.empty:
            continue
        if train["FlightDate"].max() >= future["FlightDate"].min():
            raise AssertionError("Temporal boundary overlap detected in backtest")
        folds.append((train, future))
    return folds
