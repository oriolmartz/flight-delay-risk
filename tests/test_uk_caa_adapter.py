from pathlib import Path

import pandas as pd

from src.reference.uk_caa_adapter import (
    build_uk_caa_context_dataset,
    normalize_uk_caa_punctuality_file,
)


def test_normalize_uk_caa_file_from_flexible_columns(tmp_path: Path):
    raw = tmp_path / "202407_punctuality.csv"
    pd.DataFrame(
        [
            {
                "Airline": "British Airways",
                "Departure Airport": "London Heathrow",
                "Arrival Airport": "Madrid",
                "Average Arrival Delay Mins": 12.5,
                "Percent 15min Late": 27.0,
                "Cancelled Percent": 1.2,
            }
        ]
    ).to_csv(raw, index=False)

    out = normalize_uk_caa_punctuality_file(raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["year"] == 2024
    assert row["month"] == 7
    assert row["airline"] == "BA"
    assert row["origin"] == "LHR"
    assert row["destination"] == "MAD"
    assert row["pct_flights_15min_late"] == 0.27


def test_build_uk_caa_context_dataset(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "context.csv"

    pd.DataFrame(
        [
            {
                "Carrier": "KLM",
                "Airport": "Heathrow",
                "Destination": "Amsterdam",
                "Average Delay Mins": 9.5,
                "Percentage Late": 18.0,
            }
        ]
    ).to_csv(raw_dir / "202405_full_analysis.csv", index=False)

    report = build_uk_caa_context_dataset(raw_dir, output)
    assert report.input_files == 1
    assert report.output_rows == 1
    assert output.exists()
