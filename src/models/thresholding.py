"""Decision-threshold selection utilities.

FlightRisk outputs probabilities, but the portfolio demo also needs a clear
low/moderate/high decision. The model artifact therefore stores a tuned binary
threshold selected on a validation split, not on the final test split. This
keeps reported test metrics honest while avoiding an arbitrary 0.50 cut-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    f1: float
    precision: float
    recall: float


def iter_thresholds(start: float = 0.1, stop: float = 0.9, step: float = 0.01) -> np.ndarray:
    """Return a stable grid of thresholds inclusive of both ends when possible."""
    if not 0 < start < 1 or not 0 < stop < 1 or start >= stop:
        raise ValueError("Threshold range must satisfy 0 < start < stop < 1.")
    if step <= 0:
        raise ValueError("step must be positive.")
    return np.round(np.arange(start, stop + step / 2, step), 4)


def tune_threshold_for_f1(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> ThresholdSearchResult:
    """Select the threshold that maximizes F1 on a validation set.

    Ties are resolved by choosing the threshold closest to 0.50, which avoids
    unnecessarily extreme decision boundaries when several values perform the
    same on small validation samples.
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_proba).astype(float)
    if len(y_arr) != len(p_arr):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_arr) == 0:
        raise ValueError("Cannot tune threshold on an empty validation set.")

    candidates = list(thresholds) if thresholds is not None else list(iter_thresholds())
    best: ThresholdSearchResult | None = None

    for threshold in candidates:
        pred = (p_arr >= threshold).astype(int)
        result = ThresholdSearchResult(
            threshold=float(threshold),
            f1=float(f1_score(y_arr, pred, zero_division=0)),
            precision=float(precision_score(y_arr, pred, zero_division=0)),
            recall=float(recall_score(y_arr, pred, zero_division=0)),
        )
        if best is None:
            best = result
            continue
        if result.f1 > best.f1:
            best = result
            continue
        if result.f1 == best.f1 and abs(result.threshold - 0.5) < abs(best.threshold - 0.5):
            best = result

    assert best is not None
    return best
