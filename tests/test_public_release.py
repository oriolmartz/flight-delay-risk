from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.dashboard.i18n import TEXT
from app.dashboard.streamlit_app import rank_dataframe, validate_schedule_dataframe
from app.services import prediction_service, report_service
from src.models.predict import PredictionInput
from src.version import APP_VERSION


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flight_number": "418",
                "airline": "DL",
                "origin": "JFK",
                "destination": "LAX",
                "flight_date": "2026-07-18",
                "scheduled_departure": "18:30",
                "scheduled_arrival": "21:45",
                "scheduled_duration_minutes": 375,
                "distance_miles": 2475,
            },
            {
                "flight_number": "126",
                "airline": "WN",
                "origin": "ATL",
                "destination": "BOS",
                "flight_date": "2026-07-18",
                "scheduled_departure": "08:05",
                "scheduled_arrival": "10:35",
                "scheduled_duration_minutes": 150,
                "distance_miles": 946,
            },
        ]
    )


def test_public_ui_copy_is_bilingual_and_complete():
    assert set(TEXT) == {"en", "es"}
    for lang in ("en", "es"):
        assert len(TEXT[lang]["tabs"]) == 4
        assert TEXT[lang]["batch"]["download_pdf"]
        assert TEXT[lang]["single"]["explanation"]
        assert TEXT[lang]["operations"]["performance"]


def test_natural_schedule_contract_derives_model_fields():
    result = validate_schedule_dataframe(_sample_frame())
    assert result.errors.empty
    assert len(result.payloads) == 2
    assert result.payloads[0].month == 7
    assert result.payloads[0].day_of_week == 6
    assert result.payloads[0].crs_dep_time == 1830
    assert result.prepared.iloc[0]["scheduled_departure"] == "18:30"


def test_schedule_validation_reports_row_level_errors():
    frame = _sample_frame()
    frame.loc[1, "destination"] = "ATL"
    result = validate_schedule_dataframe(frame)
    assert len(result.payloads) == 1
    assert len(result.errors) == 1
    assert int(result.errors.iloc[0]["row"]) == 3
    assert "different" in result.errors.iloc[0]["reason"]


def test_ranked_schedule_contains_context_and_relative_priority():
    ranked = rank_dataframe(_sample_frame())
    assert list(ranked["rank"]) == [1, 2]
    assert "route_rate" in ranked
    assert "route_support" in ranked
    assert "relative_exposure" in ranked
    assert "schedule_percentile" in ranked
    assert ranked.iloc[0]["delay_probability"] >= ranked.iloc[1]["delay_probability"]


def test_release_artifact_exposes_model_native_contributions():
    prediction_service.get_artifact.cache_clear()
    payload = PredictionInput(
        airline="DL",
        origin="JFK",
        destination="LAX",
        month=7,
        day_of_week=5,
        crs_dep_time=1830,
        crs_arr_time=2145,
        crs_elapsed_time=375,
        distance=2475,
    )
    result = prediction_service.predict_flight(payload)
    assert result["explanation_scale"] == "log_odds_before_calibration"
    assert result["local_contributions"]
    assert {item["direction"] for item in result["local_contributions"]} <= {"increase", "decrease"}


def test_bilingual_pdf_reports_have_valid_pdf_headers():
    payload = PredictionInput(
        airline="DL",
        origin="JFK",
        destination="LAX",
        month=7,
        day_of_week=5,
        crs_dep_time=1830,
        crs_arr_time=2145,
        crs_elapsed_time=375,
        distance=2475,
    )
    prediction = prediction_service.predict_flight(payload)
    context = prediction_service.prediction_context(payload)
    flight = {
        "airline": "DL",
        "flight_number": "418",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure": "18:30",
        "scheduled_arrival": "21:45",
        "review_label": "WATCH",
    }
    for lang in ("en", "es"):
        single_pdf = report_service.build_flight_brief_pdf(flight, prediction, context, lang=lang)
        schedule_pdf = report_service.build_schedule_brief_pdf(rank_dataframe(_sample_frame()), lang=lang)
        assert single_pdf.startswith(b"%PDF")
        assert schedule_pdf.startswith(b"%PDF")
        assert len(single_pdf) > 2000
        assert len(schedule_pdf) > 2000


def test_performance_report_is_committed_for_release():
    report = json.loads(Path("reports/performance_benchmark.json").read_text(encoding="utf-8"))
    assert report["release"] == APP_VERSION
    assert report["single_prediction_ms"]["median"] > 0
    assert report["batch_1000_ms"]["median"] > 0
