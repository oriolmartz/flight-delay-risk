from __future__ import annotations

import pandas as pd

from src.data.release_sampling import read_release_frame


def test_read_release_frame_normalizes_and_applies_training_cleaning_contract(tmp_path):
    frame = pd.DataFrame(
        {
            "YEAR": [2024, 2024, 2024, 2024],
            "MONTH": [1, 1, 1, 1],
            "DAY_OF_MONTH": [1, 2, 3, 4],
            "DAY_OF_WEEK": [1, 2, 3, 4],
            "FlightDate": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "OP_UNIQUE_CARRIER": ["AA", "AA", "AA", "AA"],
            "Origin": ["JFK", "JFK", "JFK", "JFK"],
            "Dest": ["LAX", "LAX", "LAX", "LAX"],
            "CRSDepTime": [800, 800, 800, 800],
            "CRS_ARR_TIME": [1100, 1100, 1100, 1100],
            "CRS_ELAPSED_TIME": [180, 180, 180, 180],
            "DISTANCE": [2475, 2475, 2475, 2475],
            "ARR_DEL15": [0.0, None, None, 1.0],
            "CANCELLED": [0.0, 0.0, 1.0, 0.0],
            "DIVERTED": [0.0, 0.0, 0.0, 0.0],
            "origin_temperature_c": [5.0, 6.0, 7.0, 8.0],
        }
    )
    path = tmp_path / "weather.parquet"
    frame.to_parquet(path, index=False)

    result = read_release_frame(path)

    assert list(result["ArrDel15"]) == [0, 1]
    assert list(result["Airline"]) == ["AA", "AA"]
    assert "Cancelled" not in result.columns
    assert "Diverted" not in result.columns
    assert "origin_temperature_c" in result.columns
    assert result["FlightDate"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-04",
    ]
