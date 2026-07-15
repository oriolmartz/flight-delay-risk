"""One-command real-data pipeline using the canonical preparation and training paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train_model import TrainingConfig, run_training
from src.config import (
    DATA_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    RAW_DATA_DIR,
)
from src.data.preparation import prepare_dataset
from src.models.train import PROFILE_CHOICES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BTS data and train FlightRisk.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--calibration-size", type=float, default=0.15)
    parser.add_argument("--selection-size", type=float, default=0.20)
    parser.add_argument("--selection-metric", choices=["roc_auc", "pr_auc", "f1"], default="pr_auc")
    parser.add_argument("--candidate-profile", choices=list(PROFILE_CHOICES), default="full")
    parser.add_argument("--include-gradient-boosting", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--sample-rows-per-month",
        type=int,
        default=None,
        help="Uniform sample over each complete month during preparation.",
    )
    parser.add_argument(
        "--duplicate-month-policy",
        choices=["error", "prefer-largest", "prefer-newest", "first"],
        default="error",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    args = parser.parse_args()

    if args.download:
        from scripts.download_bts_data import BTSDownloadError, download_month, parse_months

        try:
            for month in parse_months(args.months):
                download_month(
                    args.year,
                    month,
                    args.raw_dir,
                    args.sample_rows_per_month,
                )
        except BTSDownloadError as exc:
            print(str(exc))
            raise SystemExit(2) from exc

    preparation = prepare_dataset(
        args.raw_dir,
        output_path=args.processed,
        manifest_path=args.manifest,
        duplicate_month_policy=args.duplicate_month_policy,
        sample_rows_per_file=args.sample_rows_per_month,
        chunksize=args.chunksize,
    )
    result = run_training(
        TrainingConfig(
            data=preparation.output_path,
            output=args.output_model,
            data_manifest=preparation.manifest_path,
            test_size=args.test_size,
            calibration_size=args.calibration_size,
            selection_size=args.selection_size,
            selection_metric=args.selection_metric,
            include_gradient_boosting=args.include_gradient_boosting,
            max_rows=args.max_rows,
            smoothing_strength=args.smoothing_strength,
            candidate_profile=args.candidate_profile,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    logger.info("Real-data run complete: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
