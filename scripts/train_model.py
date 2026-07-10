"""Train, calibrate, evaluate and package the FlightRisk artifact.

Workflow
--------
1. Chronological development/test split.
2. Chronological model-training/validation split inside development.
3. Ordered historical encoding on model-training rows.
4. Candidate selection on validation PR-AUC (or the requested metric).
5. Post-hoc calibration and threshold tuning on validation only.
6. One final evaluation on the untouched later test period.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_PATH, REPORTS_DIR
from src.data.io import read_processed_frame
from src.data.split import split_train_test
from src.models.calibration import fit_calibration_candidates
from src.models.error_analysis import build_error_analysis, save_error_analysis
from src.models.evaluate import evaluate_model, save_reports
from src.models.experiment_tracking import MLflowRun
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import prepare_eval_frame, train_models
from src.monitoring.monitoring import build_drift_reference, save_drift_reference
from src.utils.logging import get_logger
from src.version import APP_VERSION, RELEASE_NAME

logger = get_logger(__name__)


def _choose_best_model(validation_results: dict[str, dict[str, Any]], metric: str) -> str:
    if not validation_results:
        raise ValueError("No validation results were provided.")
    return max(validation_results, key=lambda name: validation_results[name]["metrics"][metric])


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if "FlightDate" not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame["FlightDate"], errors="coerce", format="mixed")
    if dates.notna().sum() == 0:
        return None, None
    return str(dates.min().date()), str(dates.max().date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FlightRisk models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "pr_auc", "f1"],
        default="pr_auc",
    )
    parser.add_argument("--include-gradient-boosting", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument(
        "--candidate-profile",
        choices=["baseline", "linear", "full"],
        default="full",
        help="Candidate family used in this training run.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    with MLflowRun(
        "train-flightrisk", tags={"project": "FlightRisk", "version": APP_VERSION}
    ) as run:
        run.log_params(vars(args))
        logger.info("Loading processed data from %s", args.data)
        df = read_processed_frame(args.data)
        logger.info("Loaded %d rows", len(df))
        if args.max_rows is not None and len(df) > args.max_rows:
            logger.warning(
                "Using a deterministic %d-row sample across the complete time range",
                args.max_rows,
            )
            sampled = df.sample(n=args.max_rows, random_state=42).copy()
            sampled["__date"] = pd.to_datetime(sampled["FlightDate"], errors="coerce", format="mixed")
            df = (
                sampled.sort_values(["__date"], kind="stable")
                .drop(columns="__date")
                .reset_index(drop=True)
            )

        train_val_df, test_df = split_train_test(df, test_size=args.test_size)
        model_train_df, validation_df = split_train_test(
            train_val_df, test_size=args.validation_size
        )

        models, aggregates, X_train, _ = train_models(
            model_train_df.copy(),
            include_gradient_boosting=args.include_gradient_boosting,
            ordered_historical_encoding=True,
            smoothing_strength=args.smoothing_strength,
            candidate_profile=args.candidate_profile,
        )
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models if key != "main"]
        validation_results: dict[str, dict[str, Any]] = {}
        logger.info("Evaluating candidates on the later validation period")
        for key in candidate_keys:
            result = evaluate_model(
                models[key].pipeline, models[key].name, X_val, y_val, threshold=0.5
            )
            validation_results[key] = result
            run.log_metrics(result["metrics"], prefix=f"val_{models[key].name}_")
            logger.info(
                "Validation %s: PR-AUC=%.4f Lift@10=%.3f Brier=%.4f",
                models[key].name,
                result["metrics"]["pr_auc"],
                result["metrics"]["lift_at_top_10pct"],
                result["metrics"]["brier_score"],
            )

        selected_key = _choose_best_model(validation_results, args.selection_metric)
        selected_model = models[selected_key]
        logger.info("Selected %s on validation %s", selected_model.name, args.selection_metric)

        selected_val_raw = selected_model.pipeline.predict_proba(X_val)[:, 1]
        selected_calibrator, selected_calibration_candidates = fit_calibration_candidates(
            selected_val_raw, y_val
        )
        selected_val_calibrated = selected_calibrator.transform(selected_val_raw)
        threshold_result = tune_threshold_for_f1(y_val, selected_val_calibrated)
        tuned_threshold = threshold_result.threshold

        baseline_val_raw = models["baseline"].pipeline.predict_proba(X_val)[:, 1]
        baseline_calibrator, baseline_calibration_candidates = fit_calibration_candidates(
            baseline_val_raw, y_val
        )

        selected_test_results = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=tuned_threshold,
            calibrator=selected_calibrator,
            bootstrap_samples=args.bootstrap_samples,
        )
        baseline_test_results = evaluate_model(
            models["baseline"].pipeline,
            models["baseline"].name,
            X_test,
            y_test,
            threshold=0.5,
            calibrator=baseline_calibrator,
            bootstrap_samples=args.bootstrap_samples,
        )

        model_train_start, model_train_end = _date_range(model_train_df)
        validation_start, validation_end = _date_range(validation_df)
        test_start, test_end = _date_range(test_df)
        selection_summary = {
            "selected_model_key": selected_key,
            "selected_model_name": selected_model.name,
            "selection_metric": args.selection_metric,
            "validation_metrics": {
                key: validation_results[key]["metrics"] for key in sorted(validation_results)
            },
            "decision_threshold": tuned_threshold,
            "threshold_tuning": {
                "metric": "f1_on_calibrated_validation_probabilities",
                "validation_f1": threshold_result.f1,
                "validation_precision": threshold_result.precision,
                "validation_recall": threshold_result.recall,
            },
            "split_rows": {
                "model_train": len(model_train_df),
                "validation": len(validation_df),
                "test": len(test_df),
            },
            "split_dates": {
                "model_train": [model_train_start, model_train_end],
                "validation": [validation_start, validation_end],
                "test": [test_start, test_end],
            },
        }
        calibration_summary = {
            "selected_method": selected_calibrator.method,
            "selected_model_candidates": selected_calibration_candidates,
            "baseline_method": baseline_calibrator.method,
            "baseline_candidates": baseline_calibration_candidates,
            "fit_period": "validation_only",
        }
        save_reports(
            selected_test_results,
            baseline_test_results,
            out_dir=REPORTS_DIR,
            selection_summary=selection_summary,
            calibration_summary=calibration_summary,
        )

        selected_test_raw = selected_model.pipeline.predict_proba(X_test)[:, 1]
        selected_test_proba = selected_calibrator.transform(selected_test_raw)
        error_analysis = build_error_analysis(
            X_test, y_test, selected_test_proba, tuned_threshold
        )
        save_error_analysis(error_analysis, out_dir=REPORTS_DIR)
        save_drift_reference(build_drift_reference(X_train))

        metadata = build_metadata(
            model_name=selected_model.name,
            n_train=len(model_train_df),
            n_test=len(test_df),
            extra={
                "version": APP_VERSION,
                "release_name": RELEASE_NAME,
                "artifact_schema_version": "2",
                "feature_set": "ordered_smoothed_historical_rates",
                "historical_encoding": "strictly_prior_flight_date",
                "smoothing_strength": args.smoothing_strength,
                "calibration_method": selected_calibrator.method,
                "calibration_fit_period": "validation_only",
                "baseline_model_name": models["baseline"].name,
                "candidate_models": [models[key].name for key in candidate_keys],
                "selected_model_key": selected_key,
                "selection_metric": args.selection_metric,
                "validation_rows": len(validation_df),
                "decision_threshold": tuned_threshold,
                "training_data_start": model_train_start,
                "training_data_end": model_train_end,
                "validation_data_start": validation_start,
                "validation_data_end": validation_end,
                "test_data_start": test_start,
                "test_data_end": test_end,
                "monitoring": "prediction logging + PSI drift reference",
                "include_gradient_boosting": args.include_gradient_boosting,
                "candidate_profile": args.candidate_profile,
                "max_rows": args.max_rows,
            },
        )

        artifact = FlightRiskArtifact(
            pipeline=selected_model.pipeline,
            historical_aggregates=aggregates,
            metadata=metadata,
            metrics={
                "main_model": selected_test_results["metrics"],
                "baseline_model": baseline_test_results["metrics"],
                "selection": selection_summary,
                "calibration_selection": calibration_summary,
                "error_analysis": {
                    "false_positives": error_analysis["false_positives"],
                    "false_negatives": error_analysis["false_negatives"],
                },
            },
            decision_threshold=tuned_threshold,
            probability_calibrator=selected_calibrator,
        )
        artifact_path = artifact.save(args.output)
        run.log_artifact(artifact_path)
        for report_name in (
            "metrics.json",
            "feature_importance.csv",
            "error_analysis.json",
            "error_analysis.md",
        ):
            run.log_artifact(REPORTS_DIR / report_name)
        logger.info("Training complete. Artifact saved to %s", args.output)


if __name__ == "__main__":
    main()
