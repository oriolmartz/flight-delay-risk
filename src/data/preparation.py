"""Auditable, chunked preparation of the canonical FlightRisk dataset."""
from __future__ import annotations

import calendar
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATA_MANIFEST_PATH,
    DEFAULT_PROCESSED_PATH,
    PROCESSED_COLUMNS,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data.clean import CleaningReport, clean_flights_with_report
from src.data.load_data import load_raw_csv, normalize_columns
from src.data.manifest import resolve_monthly_sources, sha256_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreparationResult:
    output_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


class _ChunkWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._schema = None
        self._csv_header_written = False
        self.path.unlink(missing_ok=True)

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = frame[PROCESSED_COLUMNS].copy()
        if self.path.suffix.lower() == ".csv":
            frame.to_csv(
                self.path,
                mode="a",
                header=not self._csv_header_written,
                index=False,
            )
            self._csv_header_written = True
            return

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise RuntimeError(
                "Chunked full-data preparation requires pyarrow. Install requirements.txt."
            ) from exc

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.path,
                self._schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _merge_cleaning_reports(reports: list[CleaningReport]) -> dict[str, int]:
    keys = CleaningReport(input_rows=0).to_dict()
    return {key: int(sum(getattr(report, key) for report in reports)) for key in keys}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def prepare_dataset(
    input_dir: Path,
    *,
    output_path: Path = DEFAULT_PROCESSED_PATH,
    manifest_path: Path = DATA_MANIFEST_PATH,
    duplicate_month_policy: str = "error",
    sample_rows_per_file: int | None = None,
    chunksize: int = 100_000,
    random_seed: int = RANDOM_SEED,
) -> PreparationResult:
    """Prepare a canonical dataset without loading the full year into memory.

    When ``sample_rows_per_file`` is provided, the sample is uniform over each
    complete source file. Head truncation is intentionally unsupported.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    selected_paths, source_infos = resolve_monthly_sources(
        input_dir, duplicate_month_policy=duplicate_month_policy
    )

    config = {
        "duplicate_month_policy": duplicate_month_policy,
        "sample_rows_per_file": sample_rows_per_file,
        "chunksize": chunksize,
        "random_seed": random_seed,
        "processed_columns": PROCESSED_COLUMNS,
    }
    writer = _ChunkWriter(output_path)
    file_reports: dict[str, dict[str, Any]] = {}
    total_rows = 0
    target_sum = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    observed_dates: dict[tuple[int, int], set[str]] = {}
    schema_dtypes: dict[str, str] | None = None

    try:
        for source_path in selected_paths:
            logger.info("Preparing source %s", source_path.name)
            chunk_reports: list[CleaningReport] = []
            raw_rows = 0
            cleaned_rows = 0

            if sample_rows_per_file is not None:
                raw_chunks = [
                    load_raw_csv(
                        source_path,
                        max_rows=sample_rows_per_file,
                        chunksize=chunksize,
                        random_seed=random_seed,
                    )
                ]
            else:
                raw_chunks = (
                    normalize_columns(chunk)
                    for chunk in pd.read_csv(
                        source_path, low_memory=False, chunksize=chunksize
                    )
                )

            for raw_chunk in raw_chunks:
                raw_rows += len(raw_chunk)
                clean_chunk, report = clean_flights_with_report(raw_chunk)
                clean_chunk = clean_chunk[PROCESSED_COLUMNS]
                chunk_reports.append(report)
                cleaned_rows += len(clean_chunk)
                if clean_chunk.empty:
                    continue

                writer.write(clean_chunk)
                total_rows += len(clean_chunk)
                target_sum += int(clean_chunk[TARGET_COL].sum())
                chunk_min = clean_chunk["FlightDate"].min()
                chunk_max = clean_chunk["FlightDate"].max()
                min_date = chunk_min if min_date is None else min(min_date, chunk_min)
                max_date = chunk_max if max_date is None else max(max_date, chunk_max)
                for (year, month), dates in clean_chunk.groupby(["Year", "Month"])["FlightDate"]:
                    observed_dates.setdefault((int(year), int(month)), set()).update(
                        value.date().isoformat() for value in dates.drop_duplicates()
                    )
                if schema_dtypes is None:
                    schema_dtypes = {
                        column: str(dtype) for column, dtype in clean_chunk.dtypes.items()
                    }

            file_reports[str(source_path.resolve())] = {
                "raw_rows_read": raw_rows,
                "cleaned_rows_written": cleaned_rows,
                "cleaning": _merge_cleaning_reports(chunk_reports),
            }
    finally:
        writer.close()

    if total_rows == 0 or not output_path.exists():
        raise ValueError("Data preparation produced no rows")

    month_coverage = []
    for (year, month), dates in sorted(observed_dates.items()):
        expected_days = calendar.monthrange(year, month)[1]
        month_coverage.append(
            {
                "year": year,
                "month": month,
                "observed_days": len(dates),
                "expected_calendar_days": expected_days,
                "complete_calendar_coverage": len(dates) == expected_days,
                "first_date": min(dates),
                "last_date": max(dates),
            }
        )

    source_records = []
    selected_cleaning_totals: dict[str, int] = {}
    for info in source_infos:
        record = info.to_dict()
        record.update(file_reports.get(info.path, {}))
        source_records.append(record)
        if info.selected and "cleaning" in record:
            for key, value in record["cleaning"].items():
                selected_cleaning_totals[key] = selected_cleaning_totals.get(key, 0) + int(value)

    root = Path(__file__).resolve().parents[2]
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "input_directory": str(input_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "preparation": config,
        "preparation_config_sha256": _config_hash(config),
        "sources": source_records,
        "dataset": {
            "rows": total_rows,
            "columns": list(PROCESSED_COLUMNS),
            "dtypes": schema_dtypes or {},
            "date_start": min_date.date().isoformat() if min_date is not None else None,
            "date_end": max_date.date().isoformat() if max_date is not None else None,
            "target_positive_rate": float(target_sum / total_rows),
            "cleaning_totals": selected_cleaning_totals,
            "month_coverage": month_coverage,
            "all_selected_months_complete": bool(month_coverage)
            and all(item["complete_calendar_coverage"] for item in month_coverage),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Prepared %d rows from %d selected monthly files. Manifest: %s",
        total_rows,
        len(selected_paths),
        manifest_path,
    )
    return PreparationResult(output_path, manifest_path, manifest)
