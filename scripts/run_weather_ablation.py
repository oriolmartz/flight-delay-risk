"""Compare current FlightRisk features with point-in-time weather context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import CALENDAR_FEATURES, CORE_SCHEDULE_FEATURES, FEATURE_COLUMNS, REPORTS_DIR
from src.data.load_data import normalize_columns
from src.data.release_sampling import read_release_frame
from src.data.temporal import split_model_selection_calibration_test
from src.models.evaluate import evaluate_model
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION
from src.weather.model_features import (
    WEATHER_MODEL_FEATURES,
    complete_weather_mask,
    impute_weather_from_training,
)


def _scope_features() -> dict[str, list[str]]:
    minimal_schedule = [
        column
        for column in CORE_SCHEDULE_FEATURES + CALENDAR_FEATURES
        if column in {
            "Origin", "Dest", "Month", "DayOfWeek", "DepHour", "ArrHour",
            "Distance", "CRSElapsedTime", "Season", "IsWeekend",
            "DepHourSin", "DepHourCos", "ArrHourSin", "ArrHourCos",
            "IsFederalHoliday", "IsHolidayWindow",
        }
    ]
    return {
        "schedule_core": minimal_schedule,
        "schedule_core_plus_weather": minimal_schedule + list(WEATHER_MODEL_FEATURES),
        "baseline_current": list(FEATURE_COLUMNS),
        "baseline_plus_weather": list(FEATURE_COLUMNS) + list(WEATHER_MODEL_FEATURES),
    }


def _drop_all_missing_features(
    training_frame: pd.DataFrame,
    requested_features: list[str],
) -> tuple[list[str], list[str]]:
    """Drop features that contain no observed value in the training partition.

    The decision is fitted on model-train only, preserving the temporal contract.
    Features may reappear automatically in larger runs when train contains evidence.
    """
    usable = [
        column
        for column in requested_features
        if column in training_frame.columns and training_frame[column].notna().any()
    ]
    dropped = [column for column in requested_features if column not in usable]
    if not usable:
        raise ValueError("No usable features remain after removing all-missing columns")
    return usable, dropped


def _markdown(payload: dict) -> str:
    lines = [
        "# FlightRisk weather ablation",
        "",
        f"Release: **v{payload['release']}**",
        "",
        "All models use the same chronological partitions and the same complete-weather cohort. ",
        "Weather values are joined at T-6h and imputed from model-training medians only.",
        "",
        f"Candidate: `{payload['candidate']}`",
        "",
        "| Scope | Usable / requested | PR-AUC | Δ vs baseline | Lift@10% | Δ vs baseline | ROC-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| `{row['scope']}` | {row['usable_feature_count']} / "
            f"{row['requested_feature_count']} | {row['pr_auc']:.4f} | "
            f"{row['delta_pr_auc']:+.4f} | {row['lift_at_top_10pct']:.3f}× | "
            f"{row['delta_lift_at_top_10pct']:+.3f}× | {row['roc_auc']:.4f} | "
            f"{row['brier_score']:.4f} |"
        )
    lines.extend([
        "", "## Cohort guardrail", "",
        "This experiment estimates incremental weather value on flights with weather available at both endpoints. "
        "It does not yet claim global-network performance.",
    ])
    return "\n".join(lines)


def run_weather_ablation(data_path: Path, max_rows: int, candidate: str) -> dict:
    frame = normalize_columns(read_release_frame(data_path, max_rows=max_rows))
    required = set(WEATHER_MODEL_FEATURES) | {"ArrDel15", "FlightDate", "CRSDepTime"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Weather-enriched dataset is missing columns: {sorted(missing)}")

    frame = frame.loc[complete_weather_mask(frame)].copy()
    frame = frame.loc[pd.to_numeric(frame["ArrDel15"], errors="coerce").isin([0, 1])].copy()
    if len(frame) < 1000:
        raise ValueError(f"Complete-weather cohort is too small: {len(frame):,} rows")

    partitions = split_model_selection_calibration_test(
        frame, test_size=0.20, calibration_size=0.15, selection_size=0.20
    )
    scopes = _scope_features()
    results: list[dict] = []

    for scope, columns in scopes.items():
        X_train, y_train, aggregates = prepare_training_frame(
            partitions.model_train, feature_columns=columns
        )
        X_selection, y_selection = prepare_eval_frame(
            partitions.selection, aggregates, feature_columns=columns
        )
        usable_columns, dropped_columns = _drop_all_missing_features(X_train, columns)
        weather_columns = [
            column for column in usable_columns if column in WEATHER_MODEL_FEATURES
        ]
        if weather_columns:
            X_train, X_selection = impute_weather_from_training(
                X_train, X_selection, columns=weather_columns
            )
        pipeline = build_candidate_pipeline(candidate, feature_columns=usable_columns)
        pipeline.fit(X_train[usable_columns], y_train)
        evaluation = evaluate_model(
            pipeline, candidate, X_selection[usable_columns], y_selection, threshold=0.5
        )
        metrics = evaluation["metrics"]
        results.append({
            "scope": scope,
            "feature_count": len(usable_columns),
            "requested_feature_count": len(columns),
            "usable_feature_count": len(usable_columns),
            "requested_features": columns,
            "features": usable_columns,
            "dropped_all_missing_features": dropped_columns,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
            "lift_at_top_10pct": metrics["lift_at_top_10pct"],
        })

    baseline = next(row for row in results if row["scope"] == "baseline_current")
    schedule_core = next(row for row in results if row["scope"] == "schedule_core")
    for row in results:
        row["delta_pr_auc"] = row["pr_auc"] - baseline["pr_auc"]
        row["delta_lift_at_top_10pct"] = (
            row["lift_at_top_10pct"] - baseline["lift_at_top_10pct"]
        )
        row["delta_vs_paired_base_pr_auc"] = None
        row["delta_vs_paired_base_lift_at_top_10pct"] = None

        if row["scope"] == "schedule_core_plus_weather":
            row["delta_vs_paired_base_pr_auc"] = row["pr_auc"] - schedule_core["pr_auc"]
            row["delta_vs_paired_base_lift_at_top_10pct"] = (
                row["lift_at_top_10pct"] - schedule_core["lift_at_top_10pct"]
            )
        elif row["scope"] == "baseline_plus_weather":
            row["delta_vs_paired_base_pr_auc"] = row["pr_auc"] - baseline["pr_auc"]
            row["delta_vs_paired_base_lift_at_top_10pct"] = (
                row["lift_at_top_10pct"] - baseline["lift_at_top_10pct"]
            )

    return {
        "release": APP_VERSION,
        "candidate": candidate,
        "data": str(data_path),
        "max_rows_before_cohort_filter": max_rows,
        "cohort": "origin_weather_available == 1 and destination_weather_available == 1",
        "cohort_rows": len(frame),
        "train_rows": len(partitions.model_train),
        "selection_rows": len(partitions.selection),
        "selection_prevalence": float(pd.to_numeric(partitions.selection["ArrDel15"]).mean()),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--candidate", default="extra_trees")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "weather_ablation.json")
    args = parser.parse_args()
    payload = run_weather_ablation(args.data, args.max_rows, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "cohort_rows": payload["cohort_rows"],
        "selection_prevalence": payload["selection_prevalence"],
        "results": payload["results"],
    }, indent=2))


if __name__ == "__main__":
    main()
