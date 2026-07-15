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


def select_calibrator_on_holdout(
    raw_fit_probabilities: Any,
    y_fit: Any,
    raw_selection_probabilities: Any,
    y_selection: Any,
    *,
    methods: tuple[str, ...] = ("identity", "sigmoid", "isotonic"),
    refit_raw_probabilities: Any | None = None,
    refit_y: Any | None = None,
) -> tuple[ProbabilityCalibrator, dict[str, Any]]:
    """Select calibration method on a later holdout, then refit it.

    Candidate calibrators are fitted only on ``raw_fit_probabilities`` and are
    compared on the chronologically later ``raw_selection_probabilities``. The
    winning method is then refitted on the complete calibration period supplied
    by ``refit_*`` (or the concatenation of fit and selection blocks).
    """
    fit_p = _as_probability_array(raw_fit_probabilities)
    fit_y = np.asarray(y_fit, dtype=int).reshape(-1)
    selection_p = _as_probability_array(raw_selection_probabilities)
    selection_y = np.asarray(y_selection, dtype=int).reshape(-1)
    if len(fit_p) != len(fit_y) or len(selection_p) != len(selection_y):
        raise ValueError("Calibration probabilities and targets must have matching lengths")
    if len(fit_y) == 0 or len(selection_y) == 0:
        raise ValueError("Calibration fit and selection blocks must both be non-empty")

    candidate_metrics: dict[str, dict[str, Any]] = {}
    successful_methods: list[str] = []
    for method in methods:
        try:
            candidate = ProbabilityCalibrator(method=method).fit(fit_p, fit_y)
            candidate_metrics[method] = probability_metrics(
                selection_y, candidate.transform(selection_p)
            )
            successful_methods.append(method)
        except ValueError as exc:
            candidate_metrics[method] = {"error": str(exc)}
    if not successful_methods:
        raise ValueError("No calibration candidate could be fitted")
    selected_method = min(
        successful_methods,
        key=lambda method: candidate_metrics[method]["brier_score"],
    )

    if refit_raw_probabilities is None or refit_y is None:
        refit_p = np.concatenate([fit_p, selection_p])
        refit_targets = np.concatenate([fit_y, selection_y])
    else:
        refit_p = _as_probability_array(refit_raw_probabilities)
        refit_targets = np.asarray(refit_y, dtype=int).reshape(-1)
    selected = ProbabilityCalibrator(method=selected_method).fit(refit_p, refit_targets)
    report: dict[str, Any] = {
        "selected_method": selected_method,
        "selection_metric": "brier_score",
        "fit_rows": int(len(fit_y)),
        "selection_rows": int(len(selection_y)),
        "refit_rows": int(len(refit_targets)),
        "candidate_metrics_on_holdout": candidate_metrics,
    }
    return selected, report
