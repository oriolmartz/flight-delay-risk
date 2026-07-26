"""Normalize downloaded station files and optionally attach them to BTS flights."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.weather.airport_station_map import load_airport_station_map
from src.weather.airport_timezone_map import load_airport_timezone_map
from src.weather.parser import normalize_weather_frame
from src.weather.point_in_time_join import PointInTimeJoinConfig, attach_weather_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-input", required=True, help="CSV file or directory of NOAA CSV files")
    parser.add_argument("--canonical-output", default="data/processed/weather_hourly.parquet")
    parser.add_argument("--flights-input", help="Optional prepared BTS CSV/parquet")
    parser.add_argument("--flights-output", default="data/processed/flights_with_weather.parquet")
    parser.add_argument("--mapping", default="data/weather/airport_station_map.csv")
    parser.add_argument(
        "--timezone-mapping",
        default="data/weather/airport_timezone_map.csv",
        help="Airport-to-IANA-timezone CSV used to calculate departure cutoffs",
    )
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--max-age-hours", type=int, default=6)
    return parser.parse_args()


def _read_many(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV weather files found under {path}")
    return pd.concat((pd.read_csv(file, low_memory=False) for file in files), ignore_index=True)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    canonical = normalize_weather_frame(_read_many(Path(args.weather_input)))
    canonical_path = Path(args.canonical_output)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(canonical_path, index=False)
    print(f"Canonical observations: {len(canonical):,} -> {canonical_path}")

    if args.flights_input:
        flights = _read_table(Path(args.flights_input))
        mapping = load_airport_station_map(args.mapping)
        timezone_mapping = load_airport_timezone_map(args.timezone_mapping)
        joined = attach_weather_context(
            flights,
            canonical,
            mapping,
            PointInTimeJoinConfig(
                prediction_horizon_hours=args.horizon_hours,
                max_observation_age_hours=args.max_age_hours,
            ),
            timezone_mapping=timezone_mapping,
        )
        output = Path(args.flights_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(output, index=False)
        print(f"Flights enriched: {len(joined):,} -> {output}")


if __name__ == "__main__":
    main()
