from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.temporal import split_model_selection_calibration_test
from src.models.calibration import select_calibrator_on_holdout


def test_four_way_temporal_protocol_has_strict_boundaries():
    df = pd.DataFrame(
        {
            "FlightDate": pd.date_range("2024-01-01", periods=120, freq="D"),
            "CRSDepTime": [800] * 120,
            "ArrDel15": [0, 1] * 60,
        }
    )
    partitions = split_model_selection_calibration_test(df)
    ordered = list(partitions.as_dict().values())
    for previous, current in zip(ordered, ordered[1:]):
        assert previous["FlightDate"].max() < current["FlightDate"].min()


def test_calibrator_method_is_selected_on_later_holdout_then_refit():
    raw_fit = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.90])
    y_fit = np.array([0, 0, 0, 1, 1, 1])
    raw_selection = np.array([0.08, 0.25, 0.65, 0.85])
    y_selection = np.array([0, 0, 1, 1])

    calibrator, report = select_calibrator_on_holdout(
        raw_fit,
        y_fit,
        raw_selection,
        y_selection,
    )

    assert calibrator.method == report["selected_method"]
    assert report["fit_rows"] == 6
    assert report["selection_rows"] == 4
    assert report["refit_rows"] == 10
    assert set(report["candidate_metrics_on_holdout"]) == {
        "identity",
        "sigmoid",
        "isotonic",
    }
