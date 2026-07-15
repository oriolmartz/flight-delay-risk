"""Prepare raw BTS monthly CSVs and write an auditable processed dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_MANIFEST_PATH, DEFAULT_PROCESSED_PATH, RAW_DATA_DIR
from src.data.preparation import prepare_dataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw BTS flight data for training.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument(
        "--duplicate-month-policy",
        choices=["error", "prefer-largest", "prefer-newest", "first"],
        default="error",
        help="Fail by default instead of silently concatenating duplicate monthly exports.",
    )
    parser.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=None,
        help="Uniform sample over every complete monthly file. Never takes the first N rows.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    result = prepare_dataset(
        args.input_dir,
        output_path=args.output,
        manifest_path=args.manifest,
        duplicate_month_policy=args.duplicate_month_policy,
        sample_rows_per_file=args.sample_rows_per_file,
        chunksize=args.chunksize,
    )
    logger.info("Processed dataset: %s", result.output_path)
    logger.info("Data manifest: %s", result.manifest_path)


if __name__ == "__main__":
    main()
