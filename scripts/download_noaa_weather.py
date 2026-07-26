"""Download NOAA Global Hourly files for mapped airports."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.weather.airport_station_map import load_airport_station_map
from src.weather.downloader import download_station_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--airports", nargs="*", help="Optional IATA subset, e.g. JFK LAX")
    parser.add_argument("--mapping", default="data/weather/airport_station_map.csv")
    parser.add_argument("--output-dir", default="data/raw/weather/noaa_global_hourly")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_airport_station_map(args.mapping)
    if args.airports:
        requested = {code.upper() for code in args.airports}
        mapping = mapping[mapping["iata"].isin(requested)]
        missing = requested.difference(mapping["iata"])
        if missing:
            raise SystemExit(f"No station mapping for: {sorted(missing)}")
    for row in mapping.itertuples(index=False):
        path = download_station_year(
            row.station_id,
            args.year,
            Path(args.output_dir) / str(args.year),
            overwrite=args.overwrite,
        )
        print(f"{row.iata}: {path}")


if __name__ == "__main__":
    main()
