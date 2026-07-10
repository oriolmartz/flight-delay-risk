"""
End-to-end fallback demo that requires no BTS download.

Runs the full FlightRisk pipeline on the bundled sample file: cleaning, training,
validation model selection, threshold tuning, test evaluation, error analysis,
drift reference generation and model artifact saving.

Usage:
    python -m scripts.run_local_demo
"""
from __future__ import annotations

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SAMPLE_CSV_PATH
from src.data.clean import clean_flights
from src.data.io import write_processed_frame
from src.data.load_data import normalize_columns
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


def main() -> None:
    logger.info("=== FlightRisk local demo (bundled sample data) ===")

    with MLflowRun("local-demo", tags={"project": "FlightRisk", "source": "sample_demo"}) as run:
        raw_df = pd.read_csv(SAMPLE_CSV_PATH)
        raw_df = normalize_columns(raw_df)
        logger.info("Loaded sample data: %d rows", len(raw_df))

        clean_df = clean_flights(raw_df)

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        processed_path = PROCESSED_DATA_DIR / "flights_processed_demo.parquet"
        actual_processed_path = write_processed_frame(clean_df, processed_path)
        logger.info("Saved processed demo data to %s", actual_processed_path)

        train_val_df, test_df = split_train_test(clean_df, test_size=0.2)
        train_df, validation_df = split_train_test(train_val_df, test_size=0.2)

        models, aggregates, X_train, _ = train_models(train_df.copy())
        X_val, y_val = prepare_eval_frame(validation_df.copy(), aggregates)
        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)

        candidate_keys = [key for key in models if key != "main"]
        validation_results = {
            key: evaluate_model(models[key].pipeline, models[key].name, X_val, y_val)
            for key in candidate_keys
        }
        selected_key = max(candidate_keys, key=lambda key: validation_results[key]["metrics"]["pr_auc"])
        selected_model = models[selected_key]

        threshold_result = tune_threshold_for_f1(y_val, selected_model.pipeline.predict_proba(X_val)[:, 1])
        tuned_threshold = threshold_result.threshold

        selected_results = evaluate_model(
            selected_model.pipeline, selected_model.name, X_test, y_test, threshold=tuned_threshold
        )
        baseline_results = evaluate_model(models["baseline"].pipeline, models["baseline"].name, X_test, y_test)

        logger.info(
            "Selected %s | Test ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f | threshold=%.2f",
            selected_model.name,
            selected_results["metrics"]["roc_auc"],
            selected_results["metrics"]["pr_auc"],
            selected_results["metrics"]["f1"],
            tuned_threshold,
        )
        run.log_params({"selected_model": selected_model.name, "decision_threshold": tuned_threshold})
        run.log_metrics(selected_results["metrics"], prefix="test_selected_")

        selection_summary = {
            "selected_model_key": selected_key,
            "selected_model_name": selected_model.name,
            "selection_metric": "pr_auc",
            "validation_metrics": {key: validation_results[key]["metrics"] for key in sorted(validation_results)},
            "decision_threshold": tuned_threshold,
            "threshold_tuning": {
                "metric": "f1",
                "validation_f1": threshold_result.f1,
                "validation_precision": threshold_result.precision,
                "validation_recall": threshold_result.recall,
            },
            "split_rows": {"model_train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        }
        save_reports(selected_results, baseline_results, out_dir=REPORTS_DIR, selection_summary=selection_summary)

        test_proba = selected_model.pipeline.predict_proba(X_test)[:, 1]
        error_analysis = build_error_analysis(X_test, y_test, test_proba, tuned_threshold)
        save_error_analysis(error_analysis, out_dir=REPORTS_DIR)
        save_drift_reference(build_drift_reference(X_train))

        metadata = build_metadata(
            model_name=selected_model.name,
            n_train=len(train_df),
            n_test=len(test_df),
            extra={
                "version": "6.1.0",
                "baseline_model_name": models["baseline"].name,
                "candidate_models": [models[key].name for key in candidate_keys],
                "selected_model_key": selected_key,
                "selection_metric": "pr_auc",
                "validation_rows": len(validation_df),
                "decision_threshold": tuned_threshold,
                "source": "sample_demo",
                "monitoring": "prediction_logging + PSI drift reference",
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
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = MODELS_DIR / "flightrisk_model.joblib"
        artifact.save(artifact_path)
        run.log_artifact(artifact_path)

    logger.info("Demo complete! Model saved to %s", artifact_path)
    logger.info("Now try: python -m uvicorn app.api.main:app --reload --port 8000")
    logger.info("Or:      python -m streamlit run app/dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
