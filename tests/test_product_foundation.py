import pandas as pd

from src.data.split import time_aware_split
from src.models.train import build_baseline_pipeline
from src.version import APP_VERSION, RELEASE_NAME


def test_public_release_version_is_single_source_of_truth():
    assert APP_VERSION == "1.5.0"
    assert RELEASE_NAME == "Self-Explaining Product UI Release"


def test_public_linear_baseline_is_regularized_logistic_regression():
    model = build_baseline_pipeline().named_steps["model"]
    assert model.solver == "saga"
    assert model.class_weight == "balanced"


def test_time_split_never_shares_a_date_across_boundary():
    dates = pd.to_datetime(
        ["2024-01-01"] * 5
        + ["2024-01-02"] * 5
        + ["2024-01-03"] * 5
        + ["2024-01-04"] * 5
    )
    frame = pd.DataFrame({"FlightDate": dates, "ArrDel15": [0, 1] * 10})
    train, test = time_aware_split(frame, test_size=0.3)
    assert train["FlightDate"].max() < test["FlightDate"].min()
    assert set(train["FlightDate"]).isdisjoint(set(test["FlightDate"]))
