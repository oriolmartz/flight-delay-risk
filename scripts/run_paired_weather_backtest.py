"""Paired expanding-window backtest for the frozen ExtraTrees weather upgrade.

The experiment answers one narrow question: does point-in-time weather add value
when model family, hyperparameters, rows, chronological folds and random seed are
held fixed? It does not re-select algorithms inside each fold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import DEFAULT_PROCESSED_PATH, FEATURE_COLUMNS, REPORTS_DIR, SCHEDULE_CONTEXT_PATH
from src.data.release_sampling import read_release_frame
from src.data.temporal import make_expanding_time_folds
from src.models.evaluate import evaluate_model
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION
from src.weather.model_features import (
    WEATHER_MODEL_FEATURES,
    complete_weather_mask,
    impute_weather_from_training,
)

BASE_FEATURES = list(FEATURE_COLUMNS)
WEATHER_FEATURES = list(FEATURE_COLUMNS) + list(WEATHER_MODEL_FEATURES)


def _date_range(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return str(dates.min().date()), str(dates.max().date())


def _drop_all_missing_features(
    training_frame: pd.DataFrame,
    requested_features: list[str],
) -> tuple[list[str], list[str]]:
    usable = [
        column
        for column in requested_features
        if column in training_frame.columns and training_frame[column].notna().any()
    ]
    dropped = [column for column in requested_features if column not in usable]
    if not usable:
        raise ValueError("No usable features remain after removing all-missing columns")
    return usable, dropped


def _metric_summary(values: list[float]) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _summarize(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = ["roc_auc", "pr_auc", "brier_score", "lift_at_top_10pct"]
    summary: dict[str, Any] = {
        "folds": len(folds),
        "base": {},
        "weather": {},
        "deltas": {},
    }
    for metric in metric_names:
        summary["base"][metric] = _metric_summary(
            [float(fold["base_metrics"][metric]) for fold in folds]
        )
        summary["weather"][metric] = _metric_summary(
            [float(fold["weather_metrics"][metric]) for fold in folds]
        )
        summary["deltas"][metric] = _metric_summary(
            [float(fold["deltas"][metric]) for fold in folds]
        )

    summary["weather_wins"] = {
        "pr_auc": sum(fold["deltas"]["pr_auc"] > 0 for fold in folds),
        "lift_at_top_10pct": sum(
            fold["deltas"]["lift_at_top_10pct"] > 0 for fold in folds
        ),
        "roc_auc": sum(fold["deltas"]["roc_auc"] > 0 for fold in folds),
        "brier_score_lower_is_better": sum(
            fold["deltas"]["brier_score"] < 0 for fold in folds
        ),
    }
    return summary


def _fit_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    requested_features: list[str],
    weather_enabled: bool,
    schedule_context,
    smoothing_strength: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    X_train, y_train, aggregates = prepare_training_frame(
        train_df,
        ordered_historical_encoding=True,
        smoothing_strength=smoothing_strength,
        schedule_context=schedule_context,
        feature_columns=requested_features,
    )
    X_test, y_test = prepare_eval_frame(
        test_df,
        aggregates,
        feature_columns=requested_features,
    )
    usable, dropped = _drop_all_missing_features(X_train, requested_features)

    if weather_enabled:
        weather_columns = [column for column in usable if column in WEATHER_MODEL_FEATURES]
        X_train, X_test = impute_weather_from_training(
            X_train,
            X_test,
            columns=weather_columns,
        )

    pipeline = build_candidate_pipeline("extra_trees", feature_columns=usable)
    pipeline.fit(X_train[usable], y_train)
    evaluation = evaluate_model(
        pipeline,
        "extra_trees",
        X_test[usable],
        y_test,
        threshold=0.5,
    )
    return evaluation["metrics"], usable, dropped


def run_paired_backtest(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    min_train_fraction: float,
    smoothing_strength: float,
    schedule_context,
) -> dict[str, Any]:
    required = set(WEATHER_MODEL_FEATURES) | {"ArrDel15", "FlightDate", "CRSDepTime"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Weather-enriched dataset is missing columns: {sorted(missing)}")

    paired = frame.loc[complete_weather_mask(frame)].copy()
    paired = paired.loc[pd.to_numeric(paired["ArrDel15"], errors="coerce").isin([0, 1])].copy()
    paired = paired.sort_values(["FlightDate", "CRSDepTime"], kind="stable").reset_index(drop=True)
    if len(paired) < 1_000:
        raise ValueError(f"Complete-weather paired cohort is too small: {len(paired):,} rows")

    folds = make_expanding_time_folds(
        paired,
        n_splits=n_splits,
        min_train_fraction=min_train_fraction,
    )
    results: list[dict[str, Any]] = []

    for fold_id, (train_df, test_df) in enumerate(folds, start=1):
        print(
            f"Fold {fold_id}/{n_splits}: train={len(train_df):,}, test={len(test_df):,}",
            flush=True,
        )
        print("  Training frozen ExtraTrees baseline_current", flush=True)
        base_metrics, base_usable, base_dropped = _fit_and_evaluate(
            train_df,
            test_df,
            requested_features=BASE_FEATURES,
            weather_enabled=False,
            schedule_context=schedule_context,
            smoothing_strength=smoothing_strength,
        )
        print("  Training frozen ExtraTrees baseline_plus_weather", flush=True)
        weather_metrics, weather_usable, weather_dropped = _fit_and_evaluate(
            train_df,
            test_df,
            requested_features=WEATHER_FEATURES,
            weather_enabled=True,
            schedule_context=schedule_context,
            smoothing_strength=smoothing_strength,
        )

        deltas = {
            metric: float(weather_metrics[metric] - base_metrics[metric])
            for metric in ("roc_auc", "pr_auc", "brier_score", "lift_at_top_10pct")
        }
        train_start, train_end = _date_range(train_df)
        test_start, test_end = _date_range(test_df)
        results.append(
            {
                "fold": fold_id,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "base_feature_count": len(base_usable),
                "weather_feature_count": len(weather_usable),
                "base_dropped_all_missing_features": base_dropped,
                "weather_dropped_all_missing_features": weather_dropped,
                "base_metrics": base_metrics,
                "weather_metrics": weather_metrics,
                "deltas": deltas,
            }
        )
        print(
            "  Result: "
            f"ΔPR-AUC={deltas['pr_auc']:+.6f}, "
            f"ΔLift@10%={deltas['lift_at_top_10pct']:+.6f}, "
            f"ΔBrier={deltas['brier_score']:+.6f}",
            flush=True,
        )

    return {
        "release": APP_VERSION,
        "protocol": {
            "experiment": "paired_frozen_extra_trees_weather_backtest",
            "strategy": "expanding_window",
            "candidate": "extra_trees",
            "candidate_selection_inside_folds": False,
            "hyperparameters": "project_default_frozen_extra_trees",
            "paired_rows": "complete_weather_at_origin_and_destination",
            "same_rows_same_folds_same_seed": True,
            "n_splits": n_splits,
            "min_train_fraction": min_train_fraction,
            "historical_encoding": "fold_local_strictly_prior_flight_date",
            "weather_imputation": "training_partition_medians_only",
            "schedule_context": "complete_target_free_published_timetable",
            "smoothing_strength": smoothing_strength,
        },
        "cohort_rows": len(paired),
        "summary": _summarize(results),
        "folds": results,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Paired ExtraTrees weather backtest",
        "",
        f"Release: **v{payload['release']}**",
        "",
        "This experiment freezes the ExtraTrees model and compares the current baseline "
        "against the same model plus point-in-time weather. Every fold uses identical rows, "
        "chronological windows, hyperparameters and random seed.",
        "",
        f"Complete-weather cohort: **{payload['cohort_rows']:,} flights**",
        "",
        "| Fold | Train period | Test period | Base PR-AUC | Weather PR-AUC | ΔPR-AUC | Base Lift@10% | Weather Lift@10% | ΔLift | Base Brier | Weather Brier | ΔBrier |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in payload["folds"]:
        base = fold["base_metrics"]
        weather = fold["weather_metrics"]
        delta = fold["deltas"]
        lines.append(
            f"| {fold['fold']} | {fold['train_start']} → {fold['train_end']} | "
            f"{fold['test_start']} → {fold['test_end']} | "
            f"{base['pr_auc']:.4f} | {weather['pr_auc']:.4f} | {delta['pr_auc']:+.4f} | "
            f"{base['lift_at_top_10pct']:.3f}× | {weather['lift_at_top_10pct']:.3f}× | "
            f"{delta['lift_at_top_10pct']:+.3f}× | {base['brier_score']:.4f} | "
            f"{weather['brier_score']:.4f} | {delta['brier_score']:+.4f} |"
        )

    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Aggregate paired deltas",
            "",
            f"- Mean ΔPR-AUC: **{summary['deltas']['pr_auc']['mean']:+.4f}**",
            f"- Mean ΔLift@10%: **{summary['deltas']['lift_at_top_10pct']['mean']:+.3f}×**",
            f"- Mean ΔBrier: **{summary['deltas']['brier_score']['mean']:+.4f}** (lower is better)",
            f"- Weather PR-AUC wins: **{summary['weather_wins']['pr_auc']}/{summary['folds']} folds**",
            f"- Weather Lift@10% wins: **{summary['weather_wins']['lift_at_top_10pct']}/{summary['folds']} folds**",
            "",
            "## Interpretation guardrail",
            "",
            "The estimate applies to flights with valid point-in-time weather at both endpoints. "
            "It isolates incremental weather value; it is not a new model-selection tournament.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-train-fraction", type=float, default=0.50)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "paired_weather_backtest_extra_trees.json",
    )
    args = parser.parse_args()

    schedule_context = load_or_fit_schedule_context_from_parquet(
        args.data,
        args.schedule_context,
    )
    frame = read_release_frame(args.data, args.max_rows)
    payload = run_paired_backtest(
        frame,
        n_splits=args.n_splits,
        min_train_fraction=args.min_train_fraction,
        smoothing_strength=args.smoothing_strength,
        schedule_context=schedule_context,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
