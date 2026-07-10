"""Unit tests for feature engineering (CRS time parsing, routes, historical aggregates, leakage guard)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import FORBIDDEN_LEAKAGE_COLUMNS, TARGET_COL
from src.features.build_features import (
    add_schedule_features,
    assert_no_leakage_columns,
    build_route,
    hhmm_to_hour,
)
from src.features.historical_aggregates import HistoricalAggregates


class TestHHMMToHour:
    def test_standard_times(self):
        s = pd.Series([1830, 930, 5, 100, 2359])
        result = hhmm_to_hour(s)
        assert result.tolist() == [18, 9, 0, 1, 23]

    def test_midnight_2400_maps_to_zero(self):
        s = pd.Series([2400])
        result = hhmm_to_hour(s)
        assert result.tolist() == [0]

    def test_zero_maps_to_zero(self):
        s = pd.Series([0])
        result = hhmm_to_hour(s)
        assert result.tolist() == [0]

    def test_handles_missing_values(self):
        s = pd.Series([1830, None, 900])
        result = hhmm_to_hour(s)
        assert len(result) == 3


class TestRouteCreation:
    def test_basic_route(self):
        origin = pd.Series(["jfk", "lax"])
        dest = pd.Series(["lax", "sfo"])
        result = build_route(origin, dest)
        assert result.tolist() == ["JFK_LAX", "LAX_SFO"]

    def test_route_is_uppercase(self):
        origin = pd.Series(["jfk"])
        dest = pd.Series(["lax"])
        result = build_route(origin, dest)
        assert result.iloc[0] == "JFK_LAX"


class TestAddScheduleFeatures:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Airline": ["DL"],
                "Origin": ["JFK"],
                "Dest": ["LAX"],
                "Month": [7],
                "DayOfWeek": [6],
                "CRSDepTime": [1830],
                "CRSArrTime": [2145],
                "CRSElapsedTime": [375],
                "Distance": [2475],
                TARGET_COL: [1],
            }
        )

    def test_creates_expected_columns(self):
        df = add_schedule_features(self._sample_df())
        for col in ["DepHour", "ArrHour", "Route", "IsWeekend"]:
            assert col in df.columns

    def test_weekend_flag(self):
        df = add_schedule_features(self._sample_df())
        assert df["IsWeekend"].iloc[0] == 1  # DayOfWeek=6 -> Saturday

    def test_weekday_flag(self):
        df = self._sample_df()
        df["DayOfWeek"] = 2
        df = add_schedule_features(df)
        assert df["IsWeekend"].iloc[0] == 0

    def test_missing_airline_raises(self):
        df = self._sample_df().drop(columns=["Airline"])
        with pytest.raises(KeyError):
            add_schedule_features(df)


class TestHistoricalAggregatesFallback:
    def _train_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Airline": ["DL", "DL", "AA", "AA"],
                "Route": ["JFK_LAX", "JFK_LAX", "ORD_ATL", "ORD_ATL"],
                "Origin": ["JFK", "JFK", "ORD", "ORD"],
                "Dest": ["LAX", "LAX", "ATL", "ATL"],
                TARGET_COL: [1, 0, 0, 0],
            }
        )

    def test_fit_computes_smoothed_rates_and_exact_counts(self):
        agg = HistoricalAggregates(smoothing_strength=2.0).fit(self._train_df())
        # Global rate is 0.25. Small cohorts are intentionally contracted toward it.
        assert agg.carrier_rates["DL"] == pytest.approx((1 + 2 * 0.25) / 4)
        assert agg.carrier_rates["AA"] == pytest.approx((0 + 2 * 0.25) / 4)
        assert agg.carrier_counts["DL"] == 2
        assert agg.carrier_counts["AA"] == 2

    def test_fallback_for_unseen_carrier(self):
        agg = HistoricalAggregates().fit(self._train_df())
        unseen_df = pd.DataFrame(
            {
                "Airline": ["ZZ"],
                "Route": ["JFK_LAX"],
                "Origin": ["JFK"],
                "Dest": ["LAX"],
            }
        )
        result = agg.transform(unseen_df)
        assert result["CarrierDelayRate"].iloc[0] == pytest.approx(agg.global_fallback)

    def test_fallback_for_unseen_route_and_airports(self):
        agg = HistoricalAggregates().fit(self._train_df())
        unseen_df = pd.DataFrame(
            {
                "Airline": ["DL"],
                "Route": ["SEA_BOS"],
                "Origin": ["SEA"],
                "Dest": ["BOS"],
            }
        )
        result = agg.transform(unseen_df)
        assert result["RouteDelayRate"].iloc[0] == pytest.approx(agg.global_fallback)
        assert result["OriginDelayRate"].iloc[0] == pytest.approx(agg.global_fallback)
        assert result["DestDelayRate"].iloc[0] == pytest.approx(agg.global_fallback)

    def test_lookup_single_matches_transform(self):
        agg = HistoricalAggregates().fit(self._train_df())
        single = agg.lookup_single("DL", "JFK", "LAX", "JFK_LAX")
        assert single["CarrierDelayRate"] == pytest.approx(agg.carrier_rates["DL"])

    def test_serialization_roundtrip(self):
        agg = HistoricalAggregates().fit(self._train_df())
        restored = HistoricalAggregates.from_dict(agg.to_dict())
        assert restored.carrier_rates == agg.carrier_rates
        assert restored.global_fallback == pytest.approx(agg.global_fallback)


class TestLeakageGuard:
    def test_raises_when_forbidden_column_present(self):
        cols = ["Airline", "Origin", "Dest", "ArrDelay"]
        with pytest.raises(ValueError):
            assert_no_leakage_columns(cols)

    def test_passes_for_clean_feature_list(self):
        from src.config import FEATURE_COLUMNS

        assert_no_leakage_columns(FEATURE_COLUMNS)  # should not raise

    def test_all_forbidden_columns_individually_detected(self):
        for col in FORBIDDEN_LEAKAGE_COLUMNS:
            with pytest.raises(ValueError):
                assert_no_leakage_columns(["Airline", col])

class TestPredictiveV61Features:
    def test_v61_schedule_features_are_created(self):
        df = pd.DataFrame(
            {
                "Airline": ["DL"],
                "Origin": ["JFK"],
                "Dest": ["LAX"],
                "Month": [7],
                "DayOfWeek": [5],
                "CRSDepTime": [1830],
                "CRSArrTime": [2145],
                "CRSElapsedTime": [375],
                "Distance": [2475],
                TARGET_COL: [1],
            }
        )
        out = add_schedule_features(df)
        expected = {
            "DepPeriod",
            "ArrPeriod",
            "DistanceBand",
            "IsPeakHour",
            "IsRedEye",
            "IsLongHaul",
            "ScheduledSpeedMph",
            "DepHourSin",
            "DepHourCos",
            "CarrierRoute",
            "OriginDepHour",
        }
        assert expected.issubset(out.columns)
        assert out["IsEveningPeak"].iloc[0] == 1
        assert out["IsLongHaul"].iloc[0] == 1
