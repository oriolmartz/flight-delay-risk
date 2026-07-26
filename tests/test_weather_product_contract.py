from __future__ import annotations

import pandas as pd
import pytest

from app.services import prediction_service
from src.models.predict import PredictionInput


def _payload(flight_date: str) -> PredictionInput:
    return PredictionInput(
        airline="DL",
        origin="JFK",
        destination="LAX",
        month=int(flight_date[5:7]),
        day_of_week=1,
        crs_dep_time=1830,
        crs_arr_time=2145,
        crs_elapsed_time=375,
        distance=2475,
        flight_date=flight_date,
    )


def test_future_flight_remains_schedule_only_without_reading_historical_weather(monkeypatch):
    def fail_if_called():
        raise AssertionError("Historical weather files must not be read for a future flight")

    monkeypatch.setattr(prediction_service, "_weather_source_frames", fail_if_called)

    result = prediction_service.weather_enhanced_prediction(
        _payload("2030-07-15"),
        base_result={"delay_probability": 0.364},
    )

    assert result["mode"] == "future_schedule_only"
    assert result["available"] is False
    assert result["weather_available"] is False
    assert result["weather_delta"] is None
    assert result["reason"] == "live_forecast_feed_required"
    assert result["operational_for_future_flights"] is False
    assert result["requires_live_forecast_feed"] is True
    assert result["deployed_probability"] == pytest.approx(0.364)


def test_supported_2024_flight_can_return_paired_historical_replay(monkeypatch):
    snapshot = {
        "available": True,
        "weather_available": True,
        "origin": {"available": True},
        "destination": {"available": True},
        "feature_values": {"origin_temperature_c": 26.7},
        "prediction_cutoff_utc": "2024-07-15T16:30:00+00:00",
    }
    monkeypatch.setattr(prediction_service, "weather_snapshot", lambda payload: snapshot)
    monkeypatch.setattr(prediction_service, "get_weather_base_artifact", lambda: object())
    monkeypatch.setattr(prediction_service, "get_weather_artifact", lambda: object())

    scores = iter(
        [
            (0.329, 0.329, pd.DataFrame({"feature": [0.0]})),
            (0.344, 0.344, pd.DataFrame({"feature": [1.0]})),
        ]
    )
    monkeypatch.setattr(
        prediction_service,
        "_score_optional_artifact",
        lambda *args, **kwargs: next(scores),
    )
    monkeypatch.setattr(
        prediction_service,
        "local_model_contributions",
        lambda *args, **kwargs: [[]],
    )

    result = prediction_service.weather_enhanced_prediction(
        _payload("2024-07-15"),
        base_result={"delay_probability": 0.364},
    )

    assert result["mode"] == "historical_replay"
    assert result["available"] is True
    assert result["weather_available"] is True
    assert result["weather_delta"] == pytest.approx(0.015)
    assert result["deployed_probability"] == pytest.approx(0.364)
    assert result["interpretation"] == "historical_paired_model_diagnostic_not_causal"
