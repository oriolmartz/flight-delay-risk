import numpy as np
import pandas as pd

from src.weather.model_features import complete_weather_mask, impute_weather_from_training


def test_complete_weather_mask_requires_both_endpoints():
    frame = pd.DataFrame({
        "origin_weather_available": [1, 1, 0, 0],
        "destination_weather_available": [1, 0, 1, 0],
    })
    assert complete_weather_mask(frame).tolist() == [True, False, False, False]


def test_weather_imputation_is_fitted_on_training_only():
    train = pd.DataFrame({"origin_temperature_c": [1.0, np.nan, 3.0]})
    future = pd.DataFrame({"origin_temperature_c": [100.0, np.nan]})
    train_out, future_out = impute_weather_from_training(
        train, future, columns=["origin_temperature_c"]
    )
    assert train_out["origin_temperature_c"].tolist() == [1.0, 2.0, 3.0]
    assert future_out["origin_temperature_c"].tolist() == [100.0, 2.0]
