"""Temporal robustness, drift and rolling-performance diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.models.calibration import expected_calibration_error
from src.models.evaluate import ranking_metrics


def jensen_shannon_distance(expected: Any, actual: Any) -> float:
    """Jensen-Shannon distance for categorical distributions."""
    expected_s = pd.Series(expected, dtype="object").fillna("__MISSING__").astype(str)
    actual_s = pd.Series(actual, dtype="object").fillna("__MISSING__").astype(str)
    categories = sorted(set(expected_s.unique()) | set(actual_s.unique()))
    if not categories:
        return 0.0
    p = expected_s.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    q = actual_s.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], 1e-12, None))))

    return float(np.sqrt(max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))


def population_stability_index(expected: Any, actual: Any, bins: int = 10) -> float:
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    expected_arr = expected_arr[np.isfinite(expected_arr)]
    actual_arr = actual_arr[np.isfinite(actual_arr)]
    if len(expected_arr) == 0 or len(actual_arr) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected_arr, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        low = min(expected_arr.min(), actual_arr.min())
        high = max(expected_arr.max(), actual_arr.max())
        if low == high:
            return 0.0
        edges = np.linspace(low, high, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(expected_arr, bins=edges)
    actual_counts, _ = np.histogram(actual_arr, bins=edges)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, 1.0)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, 1.0)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_level(value: float, *, metric: str) -> str:
    if metric == "psi":
        return "high" if value >= 0.25 else "moderate" if value >= 0.10 else "low"
    if metric == "js_distance":
        return "high" if value >= 0.30 else "moderate" if value >= 0.15 else "low"
    raise ValueError(f"Unknown drift metric: {metric}")


def build_feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    severity_rank = {"low": 0, "moderate": 1, "high": 2}
    overall = "low"
    for column in numeric_features:
        if column not in reference or column not in current:
            continue
        value = population_stability_index(reference[column], current[column])
        level = drift_level(value, metric="psi")
        features[column] = {"metric": "psi", "value": value, "level": level}
        if severity_rank[level] > severity_rank[overall]:
            overall = level
    for column in categorical_features:
        if column not in reference or column not in current:
            continue
        value = jensen_shannon_distance(reference[column], current[column])
        level = drift_level(value, metric="js_distance")
        features[column] = {"metric": "js_distance", "value": value, "level": level}
        if severity_rank[level] > severity_rank[overall]:
            overall = level
    return {
        "status": overall,
        "feature_count": len(features),
        "high_drift_features": sorted(
            [name for name, values in features.items() if values["level"] == "high"]
        ),
        "moderate_drift_features": sorted(
            [name for name, values in features.items() if values["level"] == "moderate"]
        ),
        "features": features,
    }


def rolling_performance_report(
    dates: Any,
    y_true: Any,
    probabilities: Any,
    *,
    frequency: str = "ME",
    ranking_tie_breaker: Any | None = None,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="raise", format="mixed"),
            "target": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
            "tie_breaker": (
                np.asarray(ranking_tie_breaker, dtype=float)
                if ranking_tie_breaker is not None
                else np.asarray(probabilities, dtype=float)
            ),
        }
    ).sort_values("date")
    output: list[dict[str, Any]] = []
    for period, group in frame.groupby(pd.Grouper(key="date", freq=frequency)):
        if group.empty:
            continue
        y = group["target"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "period": str(period.date()),
            "start": str(group["date"].min().date()),
            "end": str(group["date"].max().date()),
            "rows": int(len(group)),
            "prevalence": float(y.mean()),
            "mean_probability": float(p.mean()),
            "brier_score": float(brier_score_loss(y, p)),
            "expected_calibration_error": float(expected_calibration_error(y, p)),
        }
        if len(np.unique(y)) > 1:
            row["roc_auc"] = float(roc_auc_score(y, p))
            row["pr_auc"] = float(average_precision_score(y, p))
        else:
            row["roc_auc"] = None
            row["pr_auc"] = None
        row.update(
            ranking_metrics(
                y,
                p,
                fractions=(0.10,),
                tie_breaker=group["tie_breaker"].to_numpy(dtype=float),
            )
        )
        output.append(row)
    return output
