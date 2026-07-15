from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.decision_policy import (
    PolicyCosts,
    evaluate_capacity_policy,
    select_top_fraction,
    threshold_for_capacity,
    tune_cost_sensitive_threshold,
)
from src.models.uncertainty import (
    block_bootstrap_metric_ci,
    paired_block_bootstrap_difference_ci,
)
from src.monitoring.robustness import (
    build_feature_drift_report,
    jensen_shannon_distance,
    rolling_performance_report,
)
from src.version import APP_VERSION


def test_layer4_release_version():
    assert APP_VERSION == "1.5.0"


def test_capacity_policy_enforces_exact_top_k_and_reports_lift():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.05, 0.90, 0.10, 0.80, 0.20, 0.70, 0.30, 0.40, 0.60, 0.50])
    mask = select_top_fraction(p, 0.20)
    assert mask.sum() == 2
    assert threshold_for_capacity(p, 0.20) == 0.8
    result = evaluate_capacity_policy(y, p, 0.20)
    assert result.selected_count == 2
    assert result.precision == 1.0
    assert result.lift > 1.0


def test_cost_sensitive_threshold_responds_to_declared_costs():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.20, 0.35, 0.45, 0.70, 0.90])
    conservative = tune_cost_sensitive_threshold(
        y, p, costs=PolicyCosts(false_positive_cost=2.0, false_negative_cost=0.2)
    )
    recall_first = tune_cost_sensitive_threshold(
        y, p, costs=PolicyCosts(false_positive_cost=0.05, false_negative_cost=3.0)
    )
    assert conservative.selected_share <= recall_first.selected_share


def test_week_block_bootstrap_and_paired_difference_preserve_time_clusters():
    dates = pd.date_range("2024-01-01", periods=56, freq="D")
    y = np.tile([0, 1], 28)
    good = np.where(y == 1, 0.8, 0.2)
    weak = np.linspace(0.2, 0.8, len(y))
    ci = block_bootstrap_metric_ci(
        y,
        good,
        dates,
        lambda a, p: float(np.mean((p >= 0.5) == a)),
        n_bootstrap=50,
        frequency="week",
    )
    assert ci["blocks"] >= 8
    assert ci["lower"] <= ci["mean"] <= ci["upper"]
    paired = paired_block_bootstrap_difference_ci(
        y,
        good,
        weak,
        dates,
        lambda a, p: float(np.mean((p >= 0.5) == a)),
        n_bootstrap=50,
        frequency="week",
    )
    assert paired["mean"] > 0


def test_drift_report_and_rolling_metrics_are_auditable():
    reference = pd.DataFrame(
        {"Distance": np.linspace(100, 1000, 100), "Airline": ["AA"] * 50 + ["DL"] * 50}
    )
    current = pd.DataFrame(
        {"Distance": np.linspace(900, 2000, 100), "Airline": ["WN"] * 100}
    )
    report = build_feature_drift_report(
        reference,
        current,
        numeric_features=["Distance"],
        categorical_features=["Airline"],
    )
    assert report["status"] in {"moderate", "high"}
    assert jensen_shannon_distance(reference["Airline"], current["Airline"]) > 0

    dates = pd.date_range("2024-10-01", periods=90, freq="D")
    y = np.tile([0, 0, 1], 30)
    p = np.where(y == 1, 0.7, 0.2)
    rows = rolling_performance_report(dates, y, p)
    assert len(rows) == 3
    assert all(row["rows"] > 0 for row in rows)
