"""Build or reuse the target-free FlightRisk schedule-context cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import DEFAULT_PROCESSED_PATH, SCHEDULE_CONTEXT_PATH
from src.features.schedule_context import ScheduleContextReference, _key


def _map_from_grouped(table: pa.Table, key_columns: list[str], occurrences: dict[int, int]) -> dict[str, float]:
    count_column = next(name for name in table.column_names if name.endswith("count_all"))
    columns = [table.column(name).to_pylist() for name in key_columns]
    values = table.column(count_column).to_numpy(zero_copy_only=False)
    output: dict[str, float] = {}
    for parts, value in zip(zip(*columns, strict=False), values, strict=False):
        denominator = max(float(occurrences.get(int(parts[0]), 52)), 1.0)
        output[_key(*parts)] = float(value / denominator)
    return output


def _resolve_parquet_column(schema_names: set[str], canonical: str, *aliases: str) -> str:
    """Return the first available parquet column for a canonical field."""
    for candidate in (canonical, *aliases):
        if candidate in schema_names:
            return candidate
    expected = ", ".join((canonical, *aliases))
    raise ValueError(
        f"Missing required parquet column for {canonical!r}. "
        f"Expected one of: {expected}. Available columns: {sorted(schema_names)}"
    )


def fit_schedule_context_from_parquet(path: Path) -> ScheduleContextReference:
    schema_names = set(pq.ParquetFile(path).schema.names)
    source_columns = {
        "FlightDate": _resolve_parquet_column(schema_names, "FlightDate", "FL_DATE"),
        "DAY_OF_WEEK": _resolve_parquet_column(schema_names, "DAY_OF_WEEK", "DayOfWeek"),
        "Airline": _resolve_parquet_column(
            schema_names, "Airline", "OP_UNIQUE_CARRIER", "Reporting_Airline"
        ),
        "Origin": _resolve_parquet_column(schema_names, "Origin", "ORIGIN"),
        "Dest": _resolve_parquet_column(schema_names, "Dest", "DEST"),
        "CRSDepTime": _resolve_parquet_column(
            schema_names, "CRSDepTime", "CRS_DEP_TIME"
        ),
        "CRSArrTime": _resolve_parquet_column(
            schema_names, "CRSArrTime", "CRS_ARR_TIME"
        ),
    }

    table = pq.read_table(path, columns=list(source_columns.values()))
    table = table.rename_columns(list(source_columns.keys()))
    dates = pd.DatetimeIndex(table.column("FlightDate").to_numpy())
    date_start, date_end = dates.min(), dates.max()
    calendar = pd.date_range(date_start.normalize(), date_end.normalize(), freq="D")
    occurrence_series = pd.Series(calendar.dayofweek + 1).value_counts()
    occurrences = {int(key): int(value) for key, value in occurrence_series.items()}

    dep = table.column("CRSDepTime").to_numpy(zero_copy_only=False).astype(np.int32)
    arr = table.column("CRSArrTime").to_numpy(zero_copy_only=False).astype(np.int32)
    dep_slot = (((dep // 100) % 24) * 60 + np.minimum(dep % 100, 59)) // 30
    arr_slot = (((arr // 100) % 24) * 60 + np.minimum(arr % 100, 59)) // 30
    route = pc.binary_join_element_wise(table.column("Origin"), table.column("Dest"), "_")
    enriched = table.append_column("DepSlot", pa.array(dep_slot.astype(np.int8)))
    enriched = enriched.append_column("ArrSlot", pa.array(arr_slot.astype(np.int8)))
    enriched = enriched.append_column("Route", route)

    def grouped(columns_: list[str]) -> pa.Table:
        return enriched.group_by(columns_).aggregate([([], "count_all")])

    reference = ScheduleContextReference(
        origin_slot_counts=_map_from_grouped(grouped(["DAY_OF_WEEK", "Origin", "DepSlot"]), ["DAY_OF_WEEK", "Origin", "DepSlot"], occurrences),
        dest_slot_counts=_map_from_grouped(grouped(["DAY_OF_WEEK", "Dest", "ArrSlot"]), ["DAY_OF_WEEK", "Dest", "ArrSlot"], occurrences),
        origin_daily=_map_from_grouped(grouped(["DAY_OF_WEEK", "Origin"]), ["DAY_OF_WEEK", "Origin"], occurrences),
        dest_daily=_map_from_grouped(grouped(["DAY_OF_WEEK", "Dest"]), ["DAY_OF_WEEK", "Dest"], occurrences),
        carrier_origin_daily=_map_from_grouped(grouped(["DAY_OF_WEEK", "Airline", "Origin"]), ["DAY_OF_WEEK", "Airline", "Origin"], occurrences),
        route_daily=_map_from_grouped(grouped(["DAY_OF_WEEK", "Route"]), ["DAY_OF_WEEK", "Route"], occurrences),
        fitted_rows=table.num_rows,
        date_start=str(date_start.date()),
        date_end=str(date_end.date()),
    )
    return reference


def load_or_fit_schedule_context(df, path: Path = SCHEDULE_CONTEXT_PATH, *, force: bool = False):
    path = Path(path)
    if path.exists() and not force:
        reference = joblib.load(path)
        if isinstance(reference, ScheduleContextReference) and reference.compatible_with(df):
            return reference
    reference = ScheduleContextReference().fit(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reference, path, compress=3)
    return reference


def load_or_fit_schedule_context_from_parquet(data_path: Path, cache_path: Path = SCHEDULE_CONTEXT_PATH, *, force: bool = False):
    cache_path = Path(cache_path)
    metadata = pq.ParquetFile(data_path).metadata
    if cache_path.exists() and not force:
        reference = joblib.load(cache_path)
        if isinstance(reference, ScheduleContextReference) and reference.fitted_rows == metadata.num_rows:
            return reference
    reference = fit_schedule_context_from_parquet(data_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reference, cache_path, compress=3)
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    reference = load_or_fit_schedule_context_from_parquet(args.data, args.output, force=args.force)
    print({"rows": reference.fitted_rows, "start": reference.date_start, "end": reference.date_end, "path": str(args.output)})


if __name__ == "__main__":
    main()

