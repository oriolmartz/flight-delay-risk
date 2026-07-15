"""Expanding-window temporal backtest with fold-local features and calibration."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import DEFAULT_PROCESSED_PATH, REPORTS_DIR, SCHEDULE_CONTEXT_PATH
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import make_expanding_time_folds
from src.models.calibration import select_calibrator_on_holdout
from src.models.evaluate import evaluate_model
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import (
    PROFILE_CHOICES,
    candidate_keys_for_profile,
    prepare_eval_frame,
    train_candidate,
    train_models,
)
from src.utils.logging import get_logger
from src.version import APP_VERSION

logger = get_logger(__name__)


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


def _date_range(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return str(dates.min().date()), str(dates.max().date())


def _markdown_report(output: dict[str, Any]) -> str:
    lines = [
        "# FlightRisk temporal backtest",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        "Every fold rebuilds target-derived features, selects the model on a later block, "
        "calibrates on a separate block and evaluates on the next unseen period.",
        "",
        "## Aggregate results",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in output["summary"].get("metrics", {}).items():
        lines.append(
            f"| `{metric}` | {values['mean']:.4f} | {values['std']:.4f} | "
            f"{values['min']:.4f} | {values['max']:.4f} |"
        )
    lines.extend(["", "## Fold evidence", ""])
    for fold in output["folds"]:
        metrics = fold["metrics"]
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Model train: {fold['train_start']} → {fold['train_end']} ({fold['train_rows']:,} rows)",
                f"- Selection: {fold['selection_start']} → {fold['selection_end']} ({fold['selection_rows']:,} rows)",
                f"- Calibration: {fold['calibration_start']} → {fold['calibration_end']} ({fold['calibration_rows']:,} rows)",
                f"- Test: {fold['test_start']} → {fold['test_end']} ({fold['test_rows']:,} rows)",
                f"- Selected model: `{fold['selected_model']}`",
                f"- Calibration: `{fold['calibration_method']}`",
                f"- PR-AUC: {metrics['pr_auc']:.4f}",
                f"- Lift@10%: {metrics['lift_at_top_10pct']:.3f}×",
                f"- Brier score: {metrics['brier_score']:.4f}",
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
    include_gradient_boosting: bool = False,
    schedule_context=None,
) -> dict[str, Any]:
    folds = make_expanding_time_folds(
        df, n_splits=n_splits, min_train_fraction=min_train_fraction
    )
    results: list[dict[str, Any]] = []

    for fold_id, (development_df, test_df) in enumerate(folds, start=1):
        pre_calibration, calibration_df = time_aware_split(development_df, test_size=0.15)
        model_train_df, selection_df = time_aware_split(pre_calibration, test_size=0.20)
        models, selection_aggregates, _, _ = train_models(
            model_train_df,
            ordered_historical_encoding=True,
            smoothing_strength=smoothing_strength,
            candidate_profile=candidate_profile,
            include_gradient_boosting=include_gradient_boosting,
            schedule_context=schedule_context,
        )
        X_selection, y_selection = prepare_eval_frame(selection_df, selection_aggregates)
        candidate_keys = candidate_keys_for_profile(
            candidate_profile,
            include_gradient_boosting=include_gradient_boosting,
        )
        validation_results = {
            key: evaluate_model(
                models[key].pipeline,
                models[key].name,
                X_selection,
                y_selection,
                threshold=0.5,
            )
            for key in candidate_keys
        }
        selected_key = max(
            candidate_keys,
            key=lambda key: validation_results[key]["metrics"][selection_metric],
        )

        refit_df = (
            pd.concat([model_train_df, selection_df], ignore_index=True)
            .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
            .reset_index(drop=True)
        )
        selected_model, aggregates, _, _ = train_candidate(
            refit_df,
            selected_key,
            smoothing_strength=smoothing_strength,
            schedule_context=schedule_context,
        )

        calibration_fit, calibration_selection = time_aware_split(
            calibration_df, test_size=0.5
        )
        X_cal_fit, y_cal_fit = prepare_eval_frame(calibration_fit, aggregates)
        X_cal_selection, y_cal_selection = prepare_eval_frame(
            calibration_selection, aggregates
        )
        X_cal_full, y_cal_full = prepare_eval_frame(calibration_df, aggregates)
        raw_fit = selected_model.pipeline.predict_proba(X_cal_fit)[:, 1]
        raw_selection = selected_model.pipeline.predict_proba(X_cal_selection)[:, 1]
        raw_full = selected_model.pipeline.predict_proba(X_cal_full)[:, 1]
        calibrator, calibration_report = select_calibrator_on_holdout(
            raw_fit,
            y_cal_fit,
            raw_selection,
            y_cal_selection,
            refit_raw_probabilities=raw_full,
            refit_y=y_cal_full,
        )
        threshold = tune_threshold_for_f1(
            y_cal_full, calibrator.transform(raw_full)
        ).threshold

        X_test, y_test = prepare_eval_frame(test_df, aggregates)
        result = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=threshold,
            calibrator=calibrator,
        )
        train_start, train_end = _date_range(model_train_df)
        selection_start, selection_end = _date_range(selection_df)
        calibration_start, calibration_end = _date_range(calibration_df)
        test_start, test_end = _date_range(test_df)
        results.append(
            {
                "fold": fold_id,
                "selected_model": selected_model.name,
                "selected_model_key": selected_key,
                "selection_metrics": {
                    key: value["metrics"] for key, value in validation_results.items()
                },
                "calibration_method": calibrator.method,
                "calibration_selection": calibration_report,
                "threshold": threshold,
                "train_rows": len(model_train_df),
                "selection_rows": len(selection_df),
                "calibration_rows": len(calibration_df),
                "test_rows": len(test_df),
                "train_start": train_start,
                "train_end": train_end,
                "selection_start": selection_start,
                "selection_end": selection_end,
                "calibration_start": calibration_start,
                "calibration_end": calibration_end,
                "test_start": test_start,
                "test_end": test_end,
                "metrics": result["metrics"],
                "raw_probability_metrics": result.get("raw_probability_metrics"),
            }
        )

    return {
        "protocol": {
            "release": APP_VERSION,
            "strategy": "expanding_window_train_selection_calibration_test",
            "n_splits": n_splits,
            "min_train_fraction": min_train_fraction,
            "selection_metric": selection_metric,
            "candidate_profile": candidate_profile,
            "candidate_scope": candidate_keys_for_profile(
                candidate_profile,
                include_gradient_boosting=include_gradient_boosting,
            ),
            "historical_encoding": "fold_local_strictly_prior_flight_date",
            "schedule_context": "complete_target_free_published_timetable",
            "recency_windows_days": [28, 90],
            "ewma_half_life_days": 28,
            "smoothing_strength": smoothing_strength,
            "calibration_selection": "later_holdout_then_refit_on_complete_calibration_block",
        },
        "summary": summarize_backtest(results),
        "folds": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlightRisk temporal backtesting.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--selection-metric", choices=["roc_auc", "pr_auc", "f1"], default="pr_auc")
    parser.add_argument("--candidate-profile", choices=list(PROFILE_CHOICES), default="full")
    parser.add_argument("--include-gradient-boosting", action="store_true")
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "temporal_backtest.json")
    args = parser.parse_args()

    schedule_context = load_or_fit_schedule_context_from_parquet(
        args.data, args.schedule_context
    )
    df = read_release_frame(args.data, args.max_rows)
    output = run_backtest(
        df,
        n_splits=args.n_splits,
        min_train_fraction=args.min_train_fraction,
        selection_metric=args.selection_metric,
        candidate_profile=args.candidate_profile,
        smoothing_strength=args.smoothing_strength,
        include_gradient_boosting=args.include_gradient_boosting,
        schedule_context=schedule_context,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown_report(output), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
