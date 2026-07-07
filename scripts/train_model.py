"""
CLI: train, select, evaluate, and save the FlightRisk inference artifact.

v4 uses explicit ML Engineering stages:
    1. model_train: fit preprocessing, historical aggregates and candidate models
    2. validation: select candidate model + tune decision threshold
    3. test: report final metrics once, using the selected threshold
    4. post-training: error analysis, drift reference and optional MLflow tracking

Usage:
    python -m scripts.train_model --data data/processed/flights_processed.parquet --output models/flightrisk_model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_PATH, REPORTS_DIR
from src.data.io import read_processed_frame
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
    """Choose the model with the highest validation metric."""
    if not validation_results:
        raise ValueError("No validation results were provided.")
    return max(validation_results, key=lambda name: validation_results[name]["metrics"][metric])


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
        help="Validation metric used to select the deployed candidate model.",
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
        help="Optional deterministic row cap after loading processed data for faster experiments.",
    )
    args = parser.parse_args()

    with MLflowRun("train-flightrisk-v6.1", tags={"project": "FlightRisk", "version": "6.1.0"}) as run:
        run.log_params(
            {
                "test_size": args.test_size,
                "validation_size": args.validation_size,
                "selection_metric": args.selection_metric,
                "include_gradient_boosting": args.include_gradient_boosting,
                "max_rows": args.max_rows,
            }
        )

        logger.info("Loading processed data from %s", args.data)
        df = read_processed_frame(args.data)
        logger.info("Loaded %d rows", len(df))
        if args.max_rows is not None and len(df) > args.max_rows:
            logger.warning(
                "Using a deterministic sample of %d rows from %d processed rows.",
                args.max_rows,
                len(df),
            )
            df = df.sample(n=args.max_rows, random_state=42).sort_index().reset_index(drop=True)

        train_val_df, test_df = split_train_test(df, test_size=args.test_size)
        model_train_df, validation_df = split_train_test(train_val_df, test_size=args.validation_size)

        models, aggregates, X_train, _ = train_models(
            model_train_df.copy(), include_gradient_boosting=args.include_gradient_boosting
        )
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models.keys() if key not in {"main"}]
        validation_results: dict[str, dict[str, Any]] = {}

        logger.info("Evaluating candidate models on validation split...")
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

        selected_test_results = evaluate_model(
            selected_model.pipeline, selected_model.name, X_test, y_test, threshold=tuned_threshold
        )
        baseline_test_results = evaluate_model(
            models["baseline"].pipeline, models["baseline"].name, X_test, y_test, threshold=0.5
        )
        run.log_metrics(selected_test_results["metrics"], prefix="test_selected_")
        run.log_metrics(baseline_test_results["metrics"], prefix="test_baseline_")

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
        save_reports(selected_test_results, baseline_test_results, out_dir=REPORTS_DIR, selection_summary=selection_summary)

        test_proba = selected_model.pipeline.predict_proba(X_test)[:, 1]
        error_analysis = build_error_analysis(X_test, y_test, test_proba, tuned_threshold)
        save_error_analysis(error_analysis, out_dir=REPORTS_DIR)

        drift_reference = build_drift_reference(X_train)
        save_drift_reference(drift_reference)

        metadata = build_metadata(
            model_name=selected_model.name,
            n_train=len(model_train_df),
            n_test=len(test_df),
            extra={
                "version": "6.1.0",
                "baseline_model_name": models["baseline"].name,
                "candidate_models": [models[key].name for key in candidate_keys],
                "selected_model_key": selected_key,
                "selection_metric": args.selection_metric,
                "validation_rows": len(validation_df),
                "decision_threshold": tuned_threshold,
                "monitoring": "prediction_logging + PSI drift reference",
                "include_gradient_boosting": args.include_gradient_boosting,
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
                "error_analysis": {
                    "false_positives": error_analysis["false_positives"],
                    "false_negatives": error_analysis["false_negatives"],
                },
            },
            decision_threshold=tuned_threshold,
        )
        artifact_path = artifact.save(args.output)
        run.log_artifact(artifact_path)
        for report_name in ["metrics.json", "feature_importance.csv", "error_analysis.json", "error_analysis.md"]:
            run.log_artifact(REPORTS_DIR / report_name)
        logger.info("Training complete. Artifact saved to %s", args.output)


if __name__ == "__main__":
    main()
