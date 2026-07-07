"""Rolling temporal backtest for FlightRisk.

This script addresses the most obvious technical critique of the baseline
training run: one train/validation/test split is useful, but temporal data
deserves repeated forward-looking evaluation.

The backtest uses expanding-window folds:
    train: all rows before a cutoff
    test:  the next chronological block

It keeps the same leakage controls as the main pipeline because historical
aggregates are fit inside each fold using training rows only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RAW_DATA_DIR, REPORTS_DIR
from src.data.clean import clean_flights
from src.data.load_data import load_raw_directory
from src.data.split import split_train_test
from src.models.evaluate import evaluate_model
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import prepare_eval_frame, train_models
from src.utils.logging import get_logger

logger = get_logger(__name__)


def make_expanding_time_folds(df: pd.DataFrame, n_splits: int = 3, min_train_fraction: float = 0.5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window chronological folds.

    Works with one month of data too by splitting within the month; with
    multiple months it naturally evaluates later periods after earlier history.
    """
    if "FlightDate" not in df.columns:
        raise KeyError("Expected FlightDate column for temporal backtesting.")

    df = df.copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df = df.dropna(subset=["FlightDate"]).sort_values("FlightDate").reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("Need at least 100 rows for a meaningful temporal backtest.")

    n = len(df)
    start_test = int(n * min_train_fraction)
    remaining = n - start_test
    if remaining <= n_splits:
        raise ValueError("Not enough rows after min_train_fraction for requested folds.")

    fold_size = remaining // n_splits
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_splits):
        test_start = start_test + i * fold_size
        test_end = start_test + (i + 1) * fold_size if i < n_splits - 1 else n
        train = df.iloc[:test_start].copy()
        test = df.iloc[test_start:test_end].copy()
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def summarize_backtest(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["roc_auc", "pr_auc", "f1", "precision_at_top_10pct", "lift_at_top_10pct"]
    summary: dict[str, Any] = {"folds": len(fold_results), "metrics": {}}
    for metric in metrics:
        values = [fold["metrics"].get(metric) for fold in fold_results if fold["metrics"].get(metric) is not None]
        if values:
            s = pd.Series(values, dtype=float)
            summary["metrics"][metric] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding-window temporal backtest on BTS real data.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--max-rows-per-month",
        type=int,
        default=None,
        help="Optional read-time row cap per monthly CSV for fast backtest smoke runs.",
    )
    parser.add_argument("--selection-metric", choices=["roc_auc", "pr_auc", "f1"], default="pr_auc")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "temporal_backtest.json")
    args = parser.parse_args()

    raw_df = load_raw_directory(args.raw_dir, max_rows_per_file=args.max_rows_per_month)
    clean_df = clean_flights(raw_df)
    if args.max_rows is not None and len(clean_df) > args.max_rows:
        clean_df = clean_df.sample(n=args.max_rows, random_state=42).sort_index().reset_index(drop=True)

    folds = make_expanding_time_folds(clean_df, n_splits=args.n_splits, min_train_fraction=args.min_train_fraction)
    results: list[dict[str, Any]] = []

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        logger.info("Backtest fold %d/%d: train=%d test=%d", fold_id, len(folds), len(train_df), len(test_df))
        model_train_df, validation_df = split_train_test(train_df, test_size=0.2)

        models, aggregates, _, _ = train_models(model_train_df.copy())
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models.keys() if key != "main"]
        validation_results = {
            key: evaluate_model(models[key].pipeline, models[key].name, X_val, y_val, threshold=0.5)
            for key in candidate_keys
        }
        selected_key = max(candidate_keys, key=lambda k: validation_results[k]["metrics"][args.selection_metric])
        selected_model = models[selected_key]

        val_proba = selected_model.pipeline.predict_proba(X_val)[:, 1]
        tuned_threshold = tune_threshold_for_f1(y_val, val_proba).threshold

        test_result = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=tuned_threshold,
            bootstrap_samples=args.bootstrap_samples,
        )

        results.append(
            {
                "fold": fold_id,
                "selected_model": selected_model.name,
                "threshold": tuned_threshold,
                "train_rows": len(model_train_df),
                "validation_rows": len(validation_df),
                "test_rows": len(test_df),
                "train_start": str(pd.to_datetime(train_df["FlightDate"]).min().date()),
                "train_end": str(pd.to_datetime(train_df["FlightDate"]).max().date()),
                "test_start": str(pd.to_datetime(test_df["FlightDate"]).min().date()),
                "test_end": str(pd.to_datetime(test_df["FlightDate"]).max().date()),
                "metrics": test_result["metrics"],
                "confidence_intervals": test_result.get("confidence_intervals"),
            }
        )

    output = {"summary": summarize_backtest(results), "folds": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved temporal backtest report to %s", args.output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
