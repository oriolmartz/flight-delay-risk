from __future__ import annotations

import pandas as pd

from src.data.load_data import load_raw_directory


def _month_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2024] * rows,
            "Month": [1] * rows,
            "DayofMonth": list(range(1, rows + 1)),
            "DayOfWeek": [1] * rows,
            "Reporting_Airline": ["AA"] * rows,
            "Origin": ["JFK"] * rows,
            "Dest": ["LAX"] * rows,
            "CRSDepTime": [800] * rows,
            "CRSArrTime": [1100] * rows,
            "CRSElapsedTime": [180] * rows,
            "Distance": [2475] * rows,
            "ArrDel15": [0] * rows,
            "Cancelled": [0] * rows,
            "Diverted": [0] * rows,
        }
    )


def test_load_raw_directory_applies_read_time_cap_per_file(tmp_path):
    _month_frame(10).to_csv(tmp_path / "bts_2024_01.csv", index=False)
    _month_frame(10).to_csv(tmp_path / "bts_2024_02.csv", index=False)

    df = load_raw_directory(tmp_path, max_rows_per_file=3)

    assert len(df) == 6
    assert set(["Airline", "Origin", "Dest", "ArrDel15"]).issubset(df.columns)
