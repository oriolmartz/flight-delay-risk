from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_schedule_features
from src.features.historical_aggregates import HistoricalAggregates
from src.models.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
    fit_calibration_candidates,
    probability_metrics,
)


def _ordered_frame() -> pd.DataFrame:
    return add_schedule_features(
        pd.DataFrame(
            {
                "FlightDate": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
                ),
                "Airline": ["DL", "DL", "DL", "DL"],
                "Origin": ["JFK"] * 4,
                "Dest": ["LAX"] * 4,
                "Month": [1] * 4,
                "DayOfWeek": [1, 1, 2, 2],
                "CRSDepTime": [900] * 4,
                "CRSArrTime": [1200] * 4,
                "CRSElapsedTime": [180] * 4,
                "Distance": [2475] * 4,
                "ArrDel15": [1, 1, 0, 0],
            }
        )
    )


def test_ordered_encoding_uses_only_strictly_earlier_dates():
    agg = HistoricalAggregates(smoothing_strength=0.0)
    encoded = agg.fit_transform_ordered(_ordered_frame())

    # First day has no history and receives the global fallback constant.
    assert encoded.loc[0, "RouteDelayRate"] == pytest.approx(0.2)
    assert encoded.loc[1, "RouteDelayRate"] == pytest.approx(0.2)
    # Both rows on day two see both day-one labels, but neither sees day-two labels.
    assert encoded.loc[2, "RouteDelayRate"] == pytest.approx(1.0)
    assert encoded.loc[3, "RouteDelayRate"] == pytest.approx(1.0)


def test_ordered_encoding_same_day_targets_do_not_change_one_another():
    first = _ordered_frame()
    second = first.copy()
    second.loc[3, "ArrDel15"] = 1

    encoded_first = HistoricalAggregates(smoothing_strength=10.0).fit_transform_ordered(first)
    encoded_second = HistoricalAggregates(smoothing_strength=10.0).fit_transform_ordered(second)

    assert encoded_first.loc[2, "RouteDelayRate"] == pytest.approx(
        encoded_second.loc[2, "RouteDelayRate"]
    )
    assert encoded_first.loc[3, "RouteDelayRate"] == pytest.approx(
        encoded_second.loc[3, "RouteDelayRate"]
    )


def test_sigmoid_calibrator_outputs_valid_probabilities():
    raw = np.array([0.05, 0.2, 0.4, 0.7, 0.9, 0.8, 0.3, 0.1])
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    calibrator = ProbabilityCalibrator(method="sigmoid").fit(raw, y)
    output = calibrator.transform(raw)
    assert np.all(output >= 0)
    assert np.all(output <= 1)
    assert output.shape == raw.shape


def test_calibration_candidate_selection_returns_metrics():
    raw = np.array([0.01, 0.05, 0.2, 0.7, 0.8, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    calibrator, candidates = fit_calibration_candidates(raw, y)
    assert calibrator.method in {"identity", "sigmoid", "isotonic"}
    assert set(candidates) == {"identity", "sigmoid", "isotonic"}
    assert all("brier_score" in values for values in candidates.values())


def test_probability_metrics_include_ece_and_brier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = probability_metrics(y, p)
    assert metrics["brier_score"] < 0.05
    assert metrics["expected_calibration_error"] == pytest.approx(
        expected_calibration_error(y, p)
    )
