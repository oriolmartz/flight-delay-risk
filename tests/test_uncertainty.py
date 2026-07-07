import numpy as np

from src.models.uncertainty import bootstrap_metric_ci, compute_metric_confidence_intervals


def test_bootstrap_metric_ci_returns_bounds():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6])

    result = bootstrap_metric_ci(y_true, y_proba, lambda y, p: float(np.mean((p >= 0.5) == y)), n_bootstrap=20)

    assert set(result) == {"mean", "lower", "upper"}
    assert result["lower"] <= result["mean"] <= result["upper"]


def test_compute_metric_confidence_intervals_contains_main_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6])

    cis = compute_metric_confidence_intervals(y_true, y_proba, threshold=0.5, n_bootstrap=20)

    assert {"roc_auc", "pr_auc", "f1"}.issubset(cis.keys())
    assert cis["pr_auc"]["lower"] <= cis["pr_auc"]["upper"]
