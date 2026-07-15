from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_raw_csv
from src.data.manifest import DuplicateMonthError, resolve_monthly_sources, sha256_file
from src.data.preparation import prepare_dataset


def _bts_frame(start: str, periods: int, *, airline: str = "AA") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "FL_DATE": dates.strftime("%Y-%m-%d"),
            "YEAR": dates.year,
            "MONTH": dates.month,
            "DAY_OF_WEEK": dates.dayofweek + 1,
            "OP_UNIQUE_CARRIER": [airline] * periods,
            "ORIGIN": ["JFK"] * periods,
            "DEST": ["LAX"] * periods,
            "CRS_DEP_TIME": [800] * periods,
            "CRS_ARR_TIME": [1100] * periods,
            "CRS_ELAPSED_TIME": [180] * periods,
            "DISTANCE": [2475] * periods,
            "ARR_DEL15": [0, 1] * (periods // 2) + ([0] if periods % 2 else []),
            "CANCELLED": [0] * periods,
            "DIVERTED": [0] * periods,
        }
    )


def test_duplicate_months_fail_before_concatenation(tmp_path: Path):
    frame = _bts_frame("2024-01-01", 3)
    frame.to_csv(tmp_path / "bts_2024_01.csv", index=False)
    frame.to_csv(tmp_path / "another_january.csv", index=False)

    with pytest.raises(DuplicateMonthError):
        resolve_monthly_sources(tmp_path, duplicate_month_policy="error")

    selected, infos = resolve_monthly_sources(
        tmp_path, duplicate_month_policy="prefer-largest"
    )
    assert len(selected) == 1
    assert sum(info.selected for info in infos) == 1


def test_uniform_read_time_sample_covers_complete_file(tmp_path: Path):
    frame = _bts_frame("2024-01-01", 100)
    path = tmp_path / "bts_2024_01.csv"
    frame.to_csv(path, index=False)

    sample = load_raw_csv(path, max_rows=20, chunksize=11, random_seed=42)

    sampled_dates = pd.to_datetime(sample["FlightDate"])
    assert len(sample) == 20
    assert sampled_dates.min() < pd.Timestamp("2024-01-20")
    assert sampled_dates.max() > pd.Timestamp("2024-03-01")


def test_prepare_dataset_writes_manifest_and_fingerprint(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _bts_frame("2024-01-01", 4).to_csv(raw_dir / "bts_2024_01.csv", index=False)
    _bts_frame("2024-02-01", 4, airline="DL").to_csv(
        raw_dir / "bts_2024_02.csv", index=False
    )
    output = tmp_path / "processed.parquet"
    manifest_path = tmp_path / "manifest.json"

    result = prepare_dataset(
        raw_dir,
        output_path=output,
        manifest_path=manifest_path,
        chunksize=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.output_path == output
    assert manifest["dataset"]["rows"] == 8
    assert len(manifest["sources"]) == 2
    assert manifest["output_sha256"] == sha256_file(output)
    assert manifest["dataset"]["date_start"] == "2024-01-01"
    assert manifest["dataset"]["date_end"] == "2024-02-04"
    assert manifest["dataset"]["all_selected_months_complete"] is False
