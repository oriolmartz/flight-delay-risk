"""Build the compact weather analytics artifact consumed by Streamlit."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import WEATHER_UI_SUMMARY_PATH
from src.data.release_sampling import read_release_frame
from src.weather.ui_analytics import build_weather_ui_summary, save_weather_ui_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weather UI summary from joined flights.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/flights_with_weather_2024.parquet"),
    )
    parser.add_argument("--output", type=Path, default=WEATHER_UI_SUMMARY_PATH)
    parser.add_argument("--top-airports", type=int, default=30)
    parser.add_argument("--min-group-support", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    frame = read_release_frame(args.data, max_rows=args.max_rows)
    cleaning_report = dict(frame.attrs.get("cleaning_report", {}))
    payload = build_weather_ui_summary(
        frame,
        top_airports=args.top_airports,
        min_group_support=args.min_group_support,
    )
    payload["cleaning_report"] = cleaning_report
    output = save_weather_ui_summary(payload, args.output)
    print(f"Saved weather UI summary to {output}")
    print(f"Rows: {payload['rows']:,}; airports: {len(payload['airport_layers']):,}")


if __name__ == "__main__":
    main()
