import numpy as np
import pandas as pd

from src.models.evaluate import ranking_metrics


def test_ranking_metrics_lift_at_top_decile():
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.95, 0.90, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13])
    metrics = ranking_metrics(y, p, fractions=(0.10,))
    assert metrics["baseline_positive_rate"] == 0.2
    assert metrics["precision_at_top_10pct"] == 1.0
    assert metrics["lift_at_top_10pct"] == 5.0
