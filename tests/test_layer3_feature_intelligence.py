from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURE_FAMILIES
from src.features.build_features import add_schedule_features
from src.features.historical_aggregates import HistoricalAggregates
from src.features.schedule_context import ScheduleContextReference


def _frame(days: int = 40, flights_per_day: int = 3) -> pd.DataFrame:
    rows = []
    for day in range(days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for flight in range(flights_per_day):
            rows.append(
                {
                    "FlightDate": date,
                    "Year": 2024,
                    "Month": date.month,
                    "DayOfWeek": date.dayofweek + 1,
                    "Airline": "DL" if flight < 2 else "AA",
                    "Origin": "JFK" if flight < 2 else "BUF",
                    "Dest": "LAX",
                    "CRSDepTime": 800 + flight * 20,
                    "CRSArrTime": 1100 + flight * 20,
                    "CRSElapsedTime": 180,
                    "Distance": 2400,
                    "ArrDel15": int((day + flight) % 5 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_layer3_schema_has_six_families_and_112_features():
    assert set(FEATURE_FAMILIES) == {
        "core_schedule", "calendar", "historical_rates", "historical_support",
        "recency", "schedule_congestion",
    }
    assert len(FEATURE_COLUMNS) == 112
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_calendar_and_minute_features_are_schedule_derived():
    row = add_schedule_features(_frame(1, 1))
    assert row.loc[0, "DepMinute"] == 480
    assert row.loc[0, "CalendarDateKnown"] == 1
    assert 0 <= row.loc[0, "DaysToNearestFederalHoliday"] <= 366
    assert row.loc[0, "Season"] == "winter"


def test_target_free_context_distinguishes_dense_and_sparse_origins():
    frame = _frame(20, 3)
    context = ScheduleContextReference().fit(frame)
    transformed = context.transform(add_schedule_features(frame.iloc[[0, 2]].copy()))
    assert transformed.iloc[0]["OriginScheduledDepartures60m"] > transformed.iloc[1]["OriginScheduledDepartures60m"]
    assert transformed.iloc[0]["OriginDailyScheduledFlights"] > transformed.iloc[1]["OriginDailyScheduledFlights"]


def test_ordered_support_uses_only_strictly_prior_dates():
    frame = add_schedule_features(_frame(3, 2))
    aggregates = HistoricalAggregates().fit_transform_ordered(frame)
    first_date = aggregates["FlightDate"].min()
    assert (aggregates.loc[aggregates["FlightDate"] == first_date, "RouteHistoryCount"] == 0).all()
    later = aggregates.loc[aggregates["FlightDate"] > first_date, "RouteHistoryCount"]
    assert (later > 0).all()


def test_recency_features_are_finite_probabilities_and_consistent_trends():
    frame = add_schedule_features(_frame())
    transformed = HistoricalAggregates().fit_transform_ordered(frame)
    rate_columns = [column for column in transformed if column.endswith(("Rate28d", "Rate90d", "RateEWMA"))]
    assert rate_columns
    values = transformed[rate_columns].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    np.testing.assert_allclose(
        transformed["RouteDelayTrend28d"],
        transformed["RouteDelayRate28d"] - transformed["RouteDelayRate"],
    )
