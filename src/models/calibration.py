"""Post-hoc probability calibration utilities for FlightRisk."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

_EPS = 1e-6


def _as_probability_array(values: Any) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float).reshape(-1), _EPS, 1.0 - _EPS)


def expected_calibration_error(
    y_true: Any,
    y_proba: Any,
    *,
    n_bins: int = 10,
) -> float:
    """Weighted absolute calibration gap over fixed-width probability bins."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = _as_probability_array(y_proba)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    total = max(len(y), 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def probability_metrics(y_true: Any, y_proba: Any) -> dict[str, float]:
    """Metrics that evaluate probability quality rather than only ranking."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = _as_probability_array(y_proba)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(y, p),
        "mean_predicted_probability": float(p.mean()),
        "observed_positive_rate": float(y.mean()),
    }


@dataclass
class ProbabilityCalibrator:
    """Serializable sigmoid or isotonic post-hoc calibrator."""

    method: str = "identity"
    estimator: Any | None = None

    @staticmethod
    def _logit(probabilities: Any) -> np.ndarray:
        p = _as_probability_array(probabilities)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, raw_probabilities: Any, y_true: Any) -> "ProbabilityCalibrator":
        y = np.asarray(y_true, dtype=int).reshape(-1)
        p = _as_probability_array(raw_probabilities)
        if self.method == "identity":
            self.estimator = None
        elif self.method == "sigmoid":
            estimator = LogisticRegression(solver="lbfgs", max_iter=1000)
            estimator.fit(self._logit(p), y)
            self.estimator = estimator
        elif self.method == "isotonic":
            estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            estimator.fit(p, y)
            self.estimator = estimator
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def transform(self, raw_probabilities: Any) -> np.ndarray:
        p = _as_probability_array(raw_probabilities)
        if self.method == "identity" or self.estimator is None:
            return p
        if self.method == "sigmoid":
            return np.asarray(self.estimator.predict_proba(self._logit(p))[:, 1], dtype=float)
        if self.method == "isotonic":
            return np.asarray(self.estimator.predict(p), dtype=float)
        raise ValueError(f"Unsupported calibration method: {self.method}")

    def fit_transform(self, raw_probabilities: Any, y_true: Any) -> np.ndarray:
        return self.fit(raw_probabilities, y_true).transform(raw_probabilities)


def fit_calibration_candidates(
    raw_probabilities: Any,
    y_true: Any,
    methods: tuple[str, ...] = ("identity", "sigmoid", "isotonic"),
) -> tuple[ProbabilityCalibrator, dict[str, dict[str, float]]]:
    """Fit candidate calibrators and select the lowest-Brier option."""
    candidates: dict[str, ProbabilityCalibrator] = {}
    metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        calibrator = ProbabilityCalibrator(method=method).fit(raw_probabilities, y_true)
        calibrated = calibrator.transform(raw_probabilities)
        candidates[method] = calibrator
        metrics[method] = probability_metrics(y_true, calibrated)
    selected = min(methods, key=lambda method: metrics[method]["brier_score"])
    return candidates[selected], metrics
