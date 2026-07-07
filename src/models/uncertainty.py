"""Uncertainty estimates for model evaluation metrics.

These helpers are intentionally lightweight: they provide bootstrap confidence
intervals for metrics computed on a held-out test set. This does not replace
proper rolling backtesting, but it gives reviewers a sense of metric stability
instead of reporting a single point estimate only.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def bootstrap_metric_ci(
    y_true,
    y_proba,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Return bootstrap mean/lower/upper confidence interval for a probability metric."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)
    if len(y) == 0:
        raise ValueError("Cannot bootstrap metrics on an empty sample.")

    rng = np.random.default_rng(random_state)
    values: list[float] = []
    n = len(y)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_sample = y[idx]
        p_sample = p[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            values.append(float(metric_fn(y_sample, p_sample)))
        except ValueError:
            continue

    if not values:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}

    arr = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1.0 - alpha)),
    }


def bootstrap_f1_ci(
    y_true,
    y_proba,
    threshold: float,
    *,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Return bootstrap confidence interval for thresholded F1."""
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
) -> dict[str, dict[str, float]]:
    """Compute CIs for the main metrics used in FlightRisk."""
    return {
        "roc_auc": bootstrap_metric_ci(
            y_true, y_proba, roc_auc_score, n_bootstrap=n_bootstrap, confidence=confidence, random_state=random_state
        ),
        "pr_auc": bootstrap_metric_ci(
            y_true,
            y_proba,
            average_precision_score,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=random_state + 1,
        ),
        "f1": bootstrap_f1_ci(
            y_true,
            y_proba,
            threshold,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            random_state=random_state + 2,
        ),
    }
