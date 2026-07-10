"""One-command real-data pipeline for FlightRisk.

This script expects real BTS CSV files in ``data/raw/``. If ``--download`` is
provided, it first attempts a best-effort official BTS monthly download with
``scripts.download_bts_data``. Manual TranStats download remains the reliable
path because BTS PREZIP URLs are not stable.

The real-data path now uses the same ML-engineering workflow as the local demo:
cleaning -> model-train/validation/test split -> candidate model selection ->
threshold tuning -> held-out test evaluation -> error analysis -> drift reference
-> model artifact.

Examples:
    # Use existing CSVs in data/raw/
    python -m scripts.run_real_data_demo

    # Faster portfolio/demo training on a deterministic subset
    python -m scripts.run_real_data_demo --max-rows 200000

    # Include the slow dense GradientBoosting candidate only when desired
    python -m scripts.run_real_data_demo --include-gradient-boosting
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_PATH, RAW_DATA_DIR, REPORTS_DIR
from src.data.clean import clean_flights
from src.data.io import read_processed_frame, write_processed_frame
from src.data.load_data import load_raw_directory
from src.data.split import split_train_test
from src.models.error_analysis import build_error_analysis, save_error_analysis
from src.models.evaluate import evaluate_model, save_reports
from src.models.experiment_tracking import MLflowRun
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import prepare_eval_frame, train_models
from src.monitoring.monitoring import build_drift_reference, save_drift_reference
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _choose_best_model(validation_results: dict[str, dict[str, Any]], metric: str) -> str:
    """Choose the candidate with the highest validation metric."""
    if not validation_results:
        raise ValueError("No validation results were provided.")
    return max(validation_results, key=lambda name: validation_results[name]["metrics"][metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FlightRisk real-data pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument(
        "--selection-metric",
        choices=["roc_auc", "pr_auc", "f1"],
        default="pr_auc",
        help="Validation metric used to select the deployed candidate model.",
    )
    parser.add_argument("--download", action="store_true", help="Attempt BTS data download before training.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--max-rows-per-month",
        type=int,
        default=None,
        help="Optional read-time row cap per monthly CSV. Works for existing files in data/raw and for downloads.",
    )
    parser.add_argument(
        "--include-gradient-boosting",
        action="store_true",
        help="Train the slow dense GradientBoostingClassifier candidate. Off by default for real BTS data.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional deterministic row cap after cleaning for faster portfolio/demo training on large BTS files.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap samples for held-out test confidence intervals. Use 0 to disable.",
    )
    args = parser.parse_args()

    if args.download:
        from scripts.download_bts_data import BTSDownloadError, download_month, parse_months

        try:
            for month in parse_months(args.months):
                download_month(args.year, month, args.raw_dir, args.max_rows_per_month)
        except BTSDownloadError as exc:
            print(str(exc))
            raise SystemExit(2) from exc

    with MLflowRun("real-data-training", tags={"project": "FlightRisk", "source": "bts_real_data"}) as run:
        run.log_params(
            {
                "test_size": args.test_size,
                "validation_size": args.validation_size,
                "selection_metric": args.selection_metric,
                "include_gradient_boosting": args.include_gradient_boosting,
                "max_rows": args.max_rows,
                "max_rows_per_month": args.max_rows_per_month,
                "bootstrap_samples": args.bootstrap_samples,
            }
        )

        logger.info("Loading real BTS CSVs from %s", args.raw_dir)
        raw_df = load_raw_directory(args.raw_dir, max_rows_per_file=args.max_rows_per_month)
        clean_df = clean_flights(raw_df)
        if args.max_rows is not None and len(clean_df) > args.max_rows:
            logger.warning(
                "Using a deterministic sample of %d rows from %d cleaned rows for faster training.",
                args.max_rows,
                len(clean_df),
            )
            clean_df = clean_df.sample(n=args.max_rows, random_state=42).sort_index().reset_index(drop=True)

        actual_processed_path = write_processed_frame(clean_df, args.processed)
        df = read_processed_frame(actual_processed_path)

        train_val_df, test_df = split_train_test(df, test_size=args.test_size)
        model_train_df, validation_df = split_train_test(train_val_df, test_size=args.validation_size)

        models, aggregates, X_train, _ = train_models(
            model_train_df.copy(), include_gradient_boosting=args.include_gradient_boosting
        )
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models.keys() if key != "main"]
        validation_results: dict[str, dict[str, Any]] = {}
        logger.info("Evaluating %d candidate models on validation split...", len(candidate_keys))
        for key in candidate_keys:
            result = evaluate_model(models[key].pipeline, models[key].name, X_val, y_val, threshold=0.5)
            validation_results[key] = result
            logger.info(
                "Validation %s: ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
                models[key].name,
                result["metrics"]["roc_auc"],
                result["metrics"]["pr_auc"],
                result["metrics"]["f1"],
            )
            run.log_metrics(result["metrics"], prefix=f"val_{models[key].name}_")

        selected_key = _choose_best_model(validation_results, args.selection_metric)
        selected_model = models[selected_key]
        logger.info("Selected model: %s using validation %s", selected_model.name, args.selection_metric)

        val_proba = selected_model.pipeline.predict_proba(X_val)[:, 1]
        threshold_result = tune_threshold_for_f1(y_val, val_proba)
        tuned_threshold = threshold_result.threshold
        run.log_params({"selected_model": selected_model.name, "decision_threshold": tuned_threshold})

        selected_results = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=tuned_threshold,
            bootstrap_samples=args.bootstrap_samples,
        )
        baseline_results = evaluate_model(
            models["baseline"].pipeline,
            models["baseline"].name,
            X_test,
            y_test,
            threshold=0.5,
            bootstrap_samples=args.bootstrap_samples,
        )
        run.log_metrics(selected_results["metrics"], prefix="test_selected_")
        run.log_metrics(baseline_results["metrics"], prefix="test_baseline_")

        selection_summary = {
            "selected_model_key": selected_key,
            "selected_model_name": selected_model.name,
            "selection_metric": args.selection_metric,
            "validation_metrics": {key: validation_results[key]["metrics"] for key in sorted(validation_results)},
            "decision_threshold": tuned_threshold,
            "threshold_tuning": {
                "metric": "f1",
                "validation_f1": threshold_result.f1,
                "validation_precision": threshold_result.precision,
                "validation_recall": threshold_result.recall,
            },
            "split_rows": {
                "model_train": len(model_train_df),
                "validation": len(validation_df),
                "test": len(test_df),
            },
        }
        save_reports(selected_results, baseline_results, out_dir=REPORTS_DIR, selection_summary=selection_summary)

        test_proba = selected_model.pipeline.predict_proba(X_test)[:, 1]
        error_analysis = build_error_analysis(X_test, y_test, test_proba, tuned_threshold)
        save_error_analysis(error_analysis, out_dir=REPORTS_DIR)
        save_drift_reference(build_drift_reference(X_train))

        metadata = build_metadata(
            model_name=selected_model.name,
            n_train=len(model_train_df),
            n_test=len(test_df),
            extra={
                "version": "6.5.0",
                "feature_set": "schedule_density_ranking_ci_backtesting",
                "baseline_model_name": models["baseline"].name,
                "candidate_models": [models[key].name for key in candidate_keys],
                "selected_model_key": selected_key,
                "selection_metric": args.selection_metric,
                "validation_rows": len(validation_df),
                "decision_threshold": tuned_threshold,
                "source": "official_bts_reporting_carrier_on_time_performance",
                "raw_dir": str(args.raw_dir),
                "processed_path": str(actual_processed_path),
                "include_gradient_boosting": args.include_gradient_boosting,
                "max_rows": args.max_rows,
                "monitoring": "prediction_logging + PSI drift reference",
                "uncertainty": f"bootstrap_ci_{args.bootstrap_samples}_samples" if args.bootstrap_samples else "disabled",
            },
        )
        artifact = FlightRiskArtifact(
            pipeline=selected_model.pipeline,
            historical_aggregates=aggregates,
            metadata=metadata,
            metrics={
                "main_model": selected_results["metrics"],
                "baseline_model": baseline_results["metrics"],
                "selection": selection_summary,
                "error_analysis": {
                    "false_positives": error_analysis["false_positives"],
                    "false_negatives": error_analysis["false_negatives"],
                },
            },
            decision_threshold=tuned_threshold,
        )
        artifact_path = artifact.save(args.output_model)
        run.log_artifact(artifact_path)
        for report_name in ["metrics.json", "feature_importance.csv", "error_analysis.json", "error_analysis.md"]:
            run.log_artifact(REPORTS_DIR / report_name)

    logger.info("Real-data pipeline complete.")
    logger.info("Model saved to %s", args.output_model)
    logger.info("Reports saved to %s", REPORTS_DIR)
    logger.info(
        "Selected %s | ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f | Lift@10=%.2f | P@10=%.3f | threshold=%.3f",
        selected_model.name,
        selected_results["metrics"]["roc_auc"],
        selected_results["metrics"]["pr_auc"],
        selected_results["metrics"]["f1"],
        selected_results["metrics"].get("lift_at_top_10pct", 0.0),
        selected_results["metrics"].get("precision_at_top_10pct", 0.0),
        tuned_threshold,
    )


if __name__ == "__main__":
    main()
