"""Uncertainty estimates for independent and time-clustered evaluation.

IID bootstrap intervals are retained for backwards compatibility. The release
reports use date/week block bootstrap because flights on the same operational
day are not independent observations.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from src.models.calibration import expected_calibration_error


def _interval(values: list[float], confidence: float) -> dict[str, float | int]:
    if not values:
        return {
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "samples": 0,
        }
    arr = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1.0 - alpha)),
        "samples": int(len(arr)),
    }


def bootstrap_metric_ci(
    y_true,
    y_proba,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int]:
    """IID bootstrap interval for a probability metric."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)
    if len(y) == 0:
        raise ValueError("Cannot bootstrap metrics on an empty sample.")
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        y_sample, p_sample = y[idx], p[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            values.append(float(metric_fn(y_sample, p_sample)))
        except ValueError:
            continue
    result = _interval(values, confidence)
    result.pop("samples", None)
    return result


def _block_indices(blocks: Any, *, frequency: str = "day") -> tuple[np.ndarray, list[np.ndarray]]:
    values = pd.DatetimeIndex(pd.to_datetime(blocks, errors="raise", format="mixed"))
    if frequency == "day":
        labels = values.normalize().to_numpy()
    elif frequency == "week":
        labels = values.to_period("W-MON").astype(str).to_numpy()
    else:
        raise ValueError("frequency must be 'day' or 'week'")
    unique = pd.unique(labels)
    indices = [np.flatnonzero(labels == block) for block in unique]
    return unique, indices


def block_bootstrap_metric_ci(
    y_true: Any,
    y_proba: Any,
    blocks: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
    frequency: str = "week",
) -> dict[str, float | int | str]:
    """Resample complete operational days/weeks instead of individual flights."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_proba, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Non-empty y_true and y_proba of equal length are required")
    _, block_rows = _block_indices(blocks, frequency=frequency)
    if len(block_rows) < 2:
        raise ValueError("At least two temporal blocks are required")
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(block_rows), size=len(block_rows))
        idx = np.concatenate([block_rows[i] for i in selected])
        y_sample, p_sample = y[idx], p[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            values.append(float(metric_fn(y_sample, p_sample)))
        except ValueError:
            continue
    result = _interval(values, confidence)
    result.update({"frequency": frequency, "blocks": int(len(block_rows))})
    return result


def paired_block_bootstrap_difference_ci(
    y_true: Any,
    first_probabilities: Any,
    second_probabilities: Any,
    blocks: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
    frequency: str = "week",
) -> dict[str, float | int | str | bool]:
    """Paired CI for metric(first) - metric(second) on identical time blocks."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    first = np.asarray(first_probabilities, dtype=float).reshape(-1)
    second = np.asarray(second_probabilities, dtype=float).reshape(-1)
    if not (len(y) == len(first) == len(second)):
        raise ValueError("All arrays must have identical lengths")
    _, block_rows = _block_indices(blocks, frequency=frequency)
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(block_rows), size=len(block_rows))
        idx = np.concatenate([block_rows[i] for i in selected])
        y_sample = y[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            values.append(
                float(metric_fn(y_sample, first[idx]) - metric_fn(y_sample, second[idx]))
            )
        except ValueError:
            continue
    result = _interval(values, confidence)
    lower, upper = float(result["lower"]), float(result["upper"])
    result.update(
        {
            "frequency": frequency,
            "blocks": int(len(block_rows)),
            "difference": "first_minus_second",
            "excludes_zero": bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0)),
        }
    )
    return result


def bootstrap_f1_ci(
    y_true,
    y_proba,
    threshold: float,
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | int]:
    def _f1(y, p):
        return f1_score(y, (p >= threshold).astype(int), zero_division=0)

    return bootstrap_metric_ci(
        y_true,
        y_proba,
        _f1,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        random_state=random_state,
    )


def compute_metric_confidence_intervals(
    y_true,
    y_proba,
    threshold: float,
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float | int]]:
    return {
        "roc_auc": bootstrap_metric_ci(
            y_true, y_proba, roc_auc_score, n_bootstrap=n_bootstrap,
            confidence=confidence, random_state=random_state,
        ),
        "pr_auc": bootstrap_metric_ci(
            y_true, y_proba, average_precision_score, n_bootstrap=n_bootstrap,
            confidence=confidence, random_state=random_state + 1,
        ),
        "f1": bootstrap_f1_ci(
            y_true, y_proba, threshold, n_bootstrap=n_bootstrap,
            confidence=confidence, random_state=random_state + 2,
        ),
    }


def _lift_at_ten_percent(y: np.ndarray, p: np.ndarray) -> float:
    prevalence = float(np.mean(y))
    if prevalence <= 0 or len(y) == 0:
        return 0.0
    k = max(1, int(round(len(y) * 0.10)))
    order = np.argsort(-p, kind="stable")
    precision = float(np.mean(y[order[:k]]))
    return float(precision / prevalence)


def compute_block_confidence_intervals(
    y_true: Any,
    y_proba: Any,
    blocks: Any,
    *,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    random_state: int = 42,
    frequency: str = "week",
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
        "pr_auc": lambda y, p: float(average_precision_score(y, p)),
        "brier_score": lambda y, p: float(brier_score_loss(y, p)),
        "expected_calibration_error": lambda y, p: float(expected_calibration_error(y, p)),
        "lift_at_top_10pct": _lift_at_ten_percent,
    }
    return {
        name: block_bootstrap_metric_ci(
            y_true,
            y_proba,
            blocks,
            fn,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=random_state + offset,
            frequency=frequency,
        )
        for offset, (name, fn) in enumerate(metrics.items())
    }
