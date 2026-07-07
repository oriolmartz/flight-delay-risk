from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.thresholding import iter_thresholds, tune_threshold_for_f1


def test_iter_thresholds_returns_grid():
    thresholds = iter_thresholds(0.2, 0.4, 0.1)
    assert thresholds.tolist() == [0.2, 0.3, 0.4]


def test_tune_threshold_for_f1_returns_valid_threshold():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    result = tune_threshold_for_f1(y_true, y_proba, thresholds=[0.3, 0.5, 0.7])
    assert 0.0 < result.threshold < 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
