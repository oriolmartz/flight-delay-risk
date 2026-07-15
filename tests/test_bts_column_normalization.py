"""Regression tests for real BTS TranStats CSV column spellings."""
from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_RAW_COLUMNS
from src.data.clean import clean_flights
from src.data.load_data import normalize_columns


def test_real_bts_upper_snake_case_columns_are_normalized():
    raw = pd.DataFrame(
        {
            "ARR_DEL15": [1.0, 0.0],
            "CRS_ARR_TIME": [2145, 1010],
            "CRS_DEP_TIME": [1830, 830],
            "CRS_ELAPSED_TIME": [375, 100],
            "Cancelled": [0.0, 0.0],
            "DayOfWeek": [1, 2],
            "Dest": ["LAX", "ATL"],
            "Distance": [2475, 760],
            "Diverted": [0.0, 0.0],
            "FL_DATE": ["2024-01-01", "2024-01-02"],
            "Month": [1, 1],
            "OP_UNIQUE_CARRIER": ["DL", "AA"],
            "Origin": ["JFK", "ORD"],
            "Year": [2024, 2024],
        }
    )

    normalized = normalize_columns(raw)

    for col in REQUIRED_RAW_COLUMNS:
        assert col in normalized.columns
    assert "Airline" in normalized.columns
    assert "FlightDate" in normalized.columns

    cleaned = clean_flights(normalized)
    assert len(cleaned) == 2
    assert cleaned["Airline"].tolist() == ["DL", "AA"]
