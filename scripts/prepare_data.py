"""
CLI: read raw BTS CSV files, clean them, and save a processed parquet file.

Usage:
    python -m scripts.prepare_data --input-dir data/raw --output data/processed/flights_processed.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_PROCESSED_PATH, RAW_DATA_DIR
from src.data.clean import clean_flights
from src.data.io import write_processed_frame
from src.data.load_data import load_raw_directory
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw BTS flight data for training.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR,
                         help="Directory containing raw BTS CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_PATH,
                         help="Output parquet path.")
    args = parser.parse_args()

    logger.info("Loading raw data from %s", args.input_dir)
    raw_df = load_raw_directory(args.input_dir)

    logger.info("Cleaning and validating data...")
    clean_df = clean_flights(raw_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    actual_output = write_processed_frame(clean_df, args.output)
    logger.info("Saved processed data (%d rows) to %s", len(clean_df), actual_output)


if __name__ == "__main__":
    main()
