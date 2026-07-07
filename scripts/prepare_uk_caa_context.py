"""Build FlightRisk European context from downloaded UK CAA punctuality CSVs.

Usage:
    python -m scripts.prepare_uk_caa_context
    python -m scripts.prepare_uk_caa_context --raw-dir data/europe/uk_caa_raw --output data/europe/europe_punctuality_context.csv

Manual workflow:
    1. Download UK CAA punctuality CSV files from the official CAA flight punctuality pages.
    2. Place the CSVs in data/europe/uk_caa_raw/.
    3. Run this script.
    4. Restart the API/dashboard.

The script writes the canonical context file consumed by the European mode.
"""
from __future__ import annotations

import argparse
import json

from src.reference.uk_caa_adapter import build_uk_caa_context_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize UK CAA punctuality CSVs into FlightRisk European context.")
    parser.add_argument("--raw-dir", default="data/europe/uk_caa_raw", help="Directory containing downloaded UK CAA CSV files.")
    parser.add_argument("--output", default="data/europe/europe_punctuality_context.csv", help="Canonical European context CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_uk_caa_context_dataset(args.raw_dir, args.output)
    print(json.dumps(report.to_dict(), indent=2))
    if report.output_rows == 0:
        print("\nNo context rows were produced. Add UK CAA punctuality CSV files to the raw directory and rerun.")


if __name__ == "__main__":
    main()
