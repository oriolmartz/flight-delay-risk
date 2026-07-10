"""Expanding-window temporal backtest for FlightRisk v1.0.

Every fold recreates historical features inside the fold, selects a candidate
on a later validation block, fits post-hoc calibration on that validation
block and evaluates once on the next unseen time block.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_PROCESSED_PATH, REPORTS_DIR
from src.data.io import read_processed_frame
from src.data.split import split_train_test
from src.models.calibration import fit_calibration_candidates
from src.models.evaluate import evaluate_model
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import prepare_eval_frame, train_models
from src.utils.logging import get_logger
from src.version import APP_VERSION

logger = get_logger(__name__)


def make_expanding_time_folds(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding folds without sharing a FlightDate boundary."""
    if "FlightDate" not in df.columns:
        raise KeyError("Expected FlightDate column for temporal backtesting.")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be between 0 and 1")

    ordered = df.copy()
    ordered["FlightDate"] = pd.to_datetime(ordered["FlightDate"], errors="coerce", format="mixed")
    ordered = ordered.dropna(subset=["FlightDate"]).sort_values("FlightDate").reset_index(drop=True)
    if len(ordered) < 100:
        raise ValueError("Need at least 100 rows for a meaningful temporal backtest.")

    unique_dates = pd.Index(ordered["FlightDate"].drop_duplicates())
    initial_date_count = max(1, int(len(unique_dates) * min_train_fraction))
    remaining_dates = len(unique_dates) - initial_date_count
    if remaining_dates < n_splits:
        raise ValueError("Not enough distinct dates for the requested number of folds.")

    date_blocks = [block for block in _split_index(unique_dates[initial_date_count:], n_splits) if len(block)]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for block in date_blocks:
        test_start = block[0]
        test_end = block[-1]
        train = ordered[ordered["FlightDate"] < test_start].copy().reset_index(drop=True)
        test = ordered[
            (ordered["FlightDate"] >= test_start) & (ordered["FlightDate"] <= test_end)
        ].copy().reset_index(drop=True)
        if train.empty or test.empty:
            continue
        if train["FlightDate"].max() >= test["FlightDate"].min():
            raise AssertionError("Temporal boundary overlap detected in backtest")
        folds.append((train, test))
    return folds


def _split_index(index: pd.Index, n_splits: int) -> list[pd.Index]:
    """Dependency-free equivalent of numpy.array_split for a pandas Index."""
    n = len(index)
    base, remainder = divmod(n, n_splits)
    blocks: list[pd.Index] = []
    start = 0
    for block_id in range(n_splits):
        size = base + (1 if block_id < remainder else 0)
        blocks.append(index[start : start + size])
        start += size
    return blocks


def summarize_backtest(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision_at_top_10pct",
        "lift_at_top_10pct",
        "brier_score",
        "expected_calibration_error",
    ]
    summary: dict[str, Any] = {
        "folds": len(fold_results),
        "metrics": {},
        "selected_models": dict(
            Counter(fold.get("selected_model", "unknown") for fold in fold_results)
        ),
        "calibration_methods": dict(
            Counter(fold.get("calibration_method", "unknown") for fold in fold_results)
        ),
    }
    for metric in metric_names:
        values = [
            fold["metrics"].get(metric)
            for fold in fold_results
            if fold["metrics"].get(metric) is not None
        ]
        if values:
            series = pd.Series(values, dtype=float)
            summary["metrics"][metric] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return summary


def _markdown_report(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# FlightRisk temporal backtest",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        "Each fold trains on earlier dates, selects and calibrates on a later validation block, then evaluates on the next unseen block.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in summary.get("metrics", {}).items():
        lines.append(
            f"| `{metric}` | {values['mean']:.4f} | {values['std']:.4f} | {values['min']:.4f} | {values['max']:.4f} |"
        )
    lines.extend(["", "## Fold evidence", ""])
    for fold in output["folds"]:
        metrics = fold["metrics"]
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Train: {fold['train_start']} → {fold['train_end']} ({fold['train_rows']:,} rows)",
                f"- Validation: {fold['validation_start']} → {fold['validation_end']} ({fold['validation_rows']:,} rows)",
                f"- Test: {fold['test_start']} → {fold['test_end']} ({fold['test_rows']:,} rows)",
                f"- Selected model: `{fold['selected_model']}`",
                f"- Calibration: `{fold['calibration_method']}`",
                f"- PR-AUC: {metrics['pr_auc']:.4f}",
                f"- Lift@10%: {metrics['lift_at_top_10pct']:.3f}×",
                f"- Brier score: {metrics['brier_score']:.4f}",
                f"- ECE: {metrics['expected_calibration_error']:.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def run_backtest(
    df: pd.DataFrame,
    *,
    n_splits: int,
    min_train_fraction: float,
    selection_metric: str,
    candidate_profile: str,
    smoothing_strength: float,
) -> dict[str, Any]:
    folds = make_expanding_time_folds(
        df, n_splits=n_splits, min_train_fraction=min_train_fraction
    )
    results: list[dict[str, Any]] = []

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        logger.info(
            "Temporal fold %d/%d: train=%d test=%d",
            fold_id,
            len(folds),
            len(train_df),
            len(test_df),
        )
        model_train_df, validation_df = split_train_test(train_df, test_size=0.2)
        models, aggregates, _, _ = train_models(
            model_train_df.copy(),
            ordered_historical_encoding=True,
            smoothing_strength=smoothing_strength,
            candidate_profile=candidate_profile,
        )
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models if key != "main"]
        validation_results = {
            key: evaluate_model(
                models[key].pipeline, models[key].name, X_val, y_val, threshold=0.5
            )
            for key in candidate_keys
        }
        selected_key = max(
            candidate_keys,
            key=lambda key: validation_results[key]["metrics"][selection_metric],
        )
        selected_model = models[selected_key]
        validation_raw = selected_model.pipeline.predict_proba(X_val)[:, 1]
        calibrator, calibration_candidates = fit_calibration_candidates(
            validation_raw, y_val
        )
        validation_probability = calibrator.transform(validation_raw)
        threshold = tune_threshold_for_f1(y_val, validation_probability).threshold
        result = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=threshold,
            calibrator=calibrator,
        )

        train_dates = pd.to_datetime(model_train_df["FlightDate"])
        val_dates = pd.to_datetime(validation_df["FlightDate"])
        test_dates = pd.to_datetime(test_df["FlightDate"])
        results.append(
            {
                "fold": fold_id,
                "selected_model": selected_model.name,
                "calibration_method": calibrator.method,
                "calibration_candidates": calibration_candidates,
                "threshold": threshold,
                "train_rows": len(model_train_df),
                "validation_rows": len(validation_df),
                "test_rows": len(test_df),
                "train_start": str(train_dates.min().date()),
                "train_end": str(train_dates.max().date()),
                "validation_start": str(val_dates.min().date()),
                "validation_end": str(val_dates.max().date()),
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                "metrics": result["metrics"],
                "raw_probability_metrics": result.get("raw_probability_metrics"),
            }
        )

    return {
        "protocol": {
            "release": APP_VERSION,
            "strategy": "expanding_window",
            "n_splits": n_splits,
            "min_train_fraction": min_train_fraction,
            "selection_metric": selection_metric,
            "candidate_profile": candidate_profile,
            "historical_encoding": "strictly_prior_flight_date",
            "smoothing_strength": smoothing_strength,
            "calibration_fit": "fold_validation_only",
        },
        "summary": summarize_backtest(results),
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlightRisk temporal backtesting.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--selection-metric", choices=["roc_auc", "pr_auc", "f1"], default="pr_auc"
    )
    parser.add_argument(
        "--candidate-profile", choices=["baseline", "linear", "full"], default="linear"
    )
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument(
        "--output", type=Path, default=REPORTS_DIR / "temporal_backtest.json"
    )
    args = parser.parse_args()

    df = read_processed_frame(args.data)
    if args.max_rows is not None and len(df) > args.max_rows:
        sampled = df.sample(n=args.max_rows, random_state=42).copy()
        sampled["__date"] = pd.to_datetime(sampled["FlightDate"], errors="coerce", format="mixed")
        df = sampled.sort_values("__date").drop(columns="__date").reset_index(drop=True)

    output = run_backtest(
        df,
        n_splits=args.n_splits,
        min_train_fraction=args.min_train_fraction,
        selection_metric=args.selection_metric,
        candidate_profile=args.candidate_profile,
        smoothing_strength=args.smoothing_strength,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown_report(output), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
