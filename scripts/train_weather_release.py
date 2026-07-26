"""Train paired frozen ExtraTrees artifacts for weather counterfactual scoring.

Two models are fitted on identical complete-weather rows and chronological
partitions:
- a schedule/history-only companion;
- the same ExtraTrees configuration plus point-in-time weather.

The public deployed artifact is not overwritten. The dashboard weather delta
uses the paired companion, so the comparison does not mix training cohorts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import (
    FEATURE_COLUMNS,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
    WEATHER_BASE_MODEL_PATH,
    WEATHER_MODEL_PATH,
)
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import split_model_selection_calibration_test
from src.models.calibration import select_calibrator_on_holdout
from src.models.evaluate import evaluate_model
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION, RELEASE_NAME
from src.weather.model_features import WEATHER_MODEL_FEATURES, complete_weather_mask

BASE_FEATURES = list(FEATURE_COLUMNS)
WEATHER_FEATURES = list(FEATURE_COLUMNS) + list(WEATHER_MODEL_FEATURES)


def _usable_features(frame: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    usable = [column for column in requested if column in frame and frame[column].notna().any()]
    dropped = [column for column in requested if column not in usable]
    if not usable:
        raise ValueError("No usable features remain")
    return usable, dropped


def _fit_medians(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = values.median()
        medians[column] = 0.0 if pd.isna(median) else float(median)
    return medians


def _apply_medians(frame: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    for column, median in medians.items():
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(median)
    return output


def _calibration_indices(calibration: pd.DataFrame) -> tuple[Any, Any]:
    index_frame = pd.DataFrame(
        {
            "FlightDate": calibration["FlightDate"].reset_index(drop=True),
            "row": range(len(calibration)),
        }
    )
    fit, selection = time_aware_split(index_frame, test_size=0.5)
    return fit["row"].astype(int).to_numpy(), selection["row"].astype(int).to_numpy()


def _fit_variant(
    *,
    name: str,
    requested_features: list[str],
    refit: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    schedule_context: Any,
    smoothing_strength: float,
    cleaning_report: dict[str, Any],
) -> tuple[FlightRiskArtifact, dict[str, Any]]:
    X_train, y_train, aggregates = prepare_training_frame(
        refit,
        ordered_historical_encoding=True,
        smoothing_strength=smoothing_strength,
        schedule_context=schedule_context,
        feature_columns=requested_features,
    )
    usable, dropped = _usable_features(X_train, requested_features)
    weather_columns = [column for column in usable if column in WEATHER_MODEL_FEATURES]
    medians = _fit_medians(X_train, weather_columns)
    X_train = _apply_medians(X_train[usable], medians)

    pipeline = build_candidate_pipeline("extra_trees", feature_columns=usable)
    pipeline.fit(X_train, y_train)

    X_calibration, y_calibration = prepare_eval_frame(
        calibration, aggregates, feature_columns=requested_features
    )
    X_calibration = _apply_medians(X_calibration[usable], medians)
    fit_idx, selection_idx = _calibration_indices(calibration)
    raw_calibration = pipeline.predict_proba(X_calibration)[:, 1]
    calibrator, calibration_report = select_calibrator_on_holdout(
        raw_calibration[fit_idx],
        y_calibration.iloc[fit_idx],
        raw_calibration[selection_idx],
        y_calibration.iloc[selection_idx],
        refit_raw_probabilities=raw_calibration,
        refit_y=y_calibration,
    )
    threshold = tune_threshold_for_f1(y_calibration, calibrator.transform(raw_calibration))

    X_test, y_test = prepare_eval_frame(test, aggregates, feature_columns=requested_features)
    X_test = _apply_medians(X_test[usable], medians)
    test_result = evaluate_model(
        pipeline,
        name,
        X_test,
        y_test,
        threshold=threshold.threshold,
        calibrator=calibrator,
    )

    metadata = build_metadata(
        model_name=name,
        n_train=len(refit),
        n_test=len(test),
        extra={
            "version": APP_VERSION,
            "release_name": RELEASE_NAME,
            "artifact_schema_version": "8",
            "selected_model_key": "extra_trees",
            "feature_columns": usable,
            "feature_set": (
                "baseline_plus_point_in_time_weather"
                if weather_columns
                else "paired_complete_weather_schedule_baseline"
            ),
            "weather_model": bool(weather_columns),
            "paired_weather_counterfactual": True,
            "weather_imputation_medians": medians,
            "dropped_all_missing_features": dropped,
            "calibration_method": calibrator.method,
            "calibration_protocol": calibration_report,
            "training_protocol": "paired_frozen_extra_trees_refit_calibration_test",
            "cleaning_report": cleaning_report,
            "paired_weather_backtest": "reports/paired_weather_backtest_extra_trees.json",
        },
    )
    artifact = FlightRiskArtifact(
        pipeline=pipeline,
        historical_aggregates=aggregates,
        feature_columns=usable,
        metadata=metadata,
        metrics={"main_model": test_result["metrics"]},
        decision_threshold=threshold.threshold,
        probability_calibrator=calibrator,
    )
    result = {
        "name": name,
        "feature_count": len(usable),
        "weather_feature_count": len(weather_columns),
        "calibration_method": calibrator.method,
        "decision_threshold": threshold.threshold,
        "test_metrics": test_result["metrics"],
    }
    return artifact, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paired frozen ExtraTrees weather artifacts.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/flights_with_weather_2024.parquet"),
    )
    parser.add_argument("--base-output", type=Path, default=WEATHER_BASE_MODEL_PATH)
    parser.add_argument("--weather-output", type=Path, default=WEATHER_MODEL_PATH)
    # Backward-compatible alias for earlier instructions.
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    args = parser.parse_args()
    if args.output is not None:
        args.weather_output = args.output

    frame = read_release_frame(args.data, max_rows=args.max_rows)
    cleaning_report = dict(frame.attrs.get("cleaning_report", {}))
    frame = frame.loc[complete_weather_mask(frame)].copy()
    frame = frame.sort_values(["FlightDate", "CRSDepTime"], kind="stable").reset_index(drop=True)
    if len(frame) < 10_000:
        raise ValueError(f"Complete-weather cohort too small: {len(frame):,}")

    schedule_context = load_or_fit_schedule_context_from_parquet(args.data, args.schedule_context)
    partitions = split_model_selection_calibration_test(frame)
    refit = pd.concat([partitions.model_train, partitions.selection], ignore_index=True)
    refit = refit.sort_values(["FlightDate", "CRSDepTime"], kind="stable").reset_index(drop=True)

    print("Training paired schedule-only ExtraTrees companion", flush=True)
    base_artifact, base_result = _fit_variant(
        name="extra_trees_weather_paired_base",
        requested_features=BASE_FEATURES,
        refit=refit,
        calibration=partitions.calibration,
        test=partitions.test,
        schedule_context=schedule_context,
        smoothing_strength=args.smoothing_strength,
        cleaning_report=cleaning_report,
    )
    base_path = base_artifact.save(args.base_output)

    print("Training paired ExtraTrees plus weather", flush=True)
    weather_artifact, weather_result = _fit_variant(
        name="extra_trees_weather",
        requested_features=WEATHER_FEATURES,
        refit=refit,
        calibration=partitions.calibration,
        test=partitions.test,
        schedule_context=schedule_context,
        smoothing_strength=args.smoothing_strength,
        cleaning_report=cleaning_report,
    )
    weather_path = weather_artifact.save(args.weather_output)

    deltas = {
        metric: float(weather_result["test_metrics"][metric] - base_result["test_metrics"][metric])
        for metric in ("roc_auc", "pr_auc", "brier_score", "lift_at_top_10pct")
    }
    report = {
        "cohort_rows": len(frame),
        "train_rows": len(refit),
        "test_rows": len(partitions.test),
        "base_artifact": str(base_path),
        "weather_artifact": str(weather_path),
        "base": base_result,
        "weather": weather_result,
        "test_deltas_weather_minus_base": deltas,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "weather_release.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
