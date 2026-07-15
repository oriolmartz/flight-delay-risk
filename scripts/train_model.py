"""Canonical FlightRisk training orchestrator.

Protocol
--------
1. Model candidates are trained on the earliest block.
2. Candidate selection happens on a later, untouched block.
3. The winning family is refitted on train + selection.
4. Calibration method and decision threshold are learned on a later calibration block.
5. The final test block is evaluated once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import (
    DATA_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    FEATURE_FAMILIES,
    RANDOM_SEED,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
)
from src.data.manifest import sha256_file
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import TemporalPartitions, split_model_selection_calibration_test
from src.models.calibration import select_calibrator_on_holdout
from src.models.error_analysis import build_error_analysis, save_error_analysis
from src.models.evaluate import evaluate_model, save_reports
from src.models.experiment_tracking import MLflowRun
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.thresholding import tune_threshold_for_f1
from src.models.train import (
    PROFILE_CHOICES,
    candidate_keys_for_profile,
    prepare_eval_frame,
    train_candidate,
    train_models,
)
from src.monitoring.monitoring import build_drift_reference, save_drift_reference
from src.utils.logging import get_logger
from src.version import APP_VERSION, RELEASE_NAME

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    data: Path = DEFAULT_PROCESSED_PATH
    output: Path = DEFAULT_MODEL_PATH
    data_manifest: Path = DATA_MANIFEST_PATH
    schedule_context: Path = SCHEDULE_CONTEXT_PATH
    test_size: float = 0.20
    calibration_size: float = 0.15
    selection_size: float = 0.20
    selection_metric: str = "pr_auc"
    include_gradient_boosting: bool = False
    max_rows: int | None = None
    smoothing_strength: float = 50.0
    candidate_profile: str = "full"
    bootstrap_samples: int = 0


def _choose_best_model(validation_results: dict[str, dict[str, Any]], metric: str) -> str:
    if not validation_results:
        raise ValueError("No validation results were provided")
    return max(validation_results, key=lambda name: validation_results[name]["metrics"][metric])


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(frame["FlightDate"], errors="coerce", format="mixed")
    if dates.notna().sum() == 0:
        return None, None
    return str(dates.min().date()), str(dates.max().date())


def _sample_across_months(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Deterministic proportional sample that preserves every observed month."""
    if n_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(df) <= n_rows:
        return df.copy()
    frame = df.copy()
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    month_key = frame["FlightDate"].dt.to_period("M")
    counts = month_key.value_counts().sort_index()
    allocations = (counts / counts.sum() * n_rows).apply(lambda value: max(1, int(value)))
    while allocations.sum() > n_rows:
        key = allocations[allocations > 1].idxmax()
        allocations.loc[key] -= 1
    while allocations.sum() < n_rows:
        key = (counts - allocations).idxmax()
        allocations.loc[key] += 1

    samples = []
    for offset, (month, allocation) in enumerate(allocations.items()):
        group = frame.loc[month_key == month]
        samples.append(
            group.sample(
                n=min(int(allocation), len(group)),
                random_state=RANDOM_SEED + offset,
            )
        )
    return (
        pd.concat(samples, ignore_index=True)
        .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
        .reset_index(drop=True)
    )


def _split_calibration_block(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return time_aware_split(calibration, test_size=0.5)


def _fit_calibrator_and_threshold(model, aggregates, calibration: pd.DataFrame) -> tuple[Any, Any, dict]:
    calibration_fit, calibration_selection = _split_calibration_block(calibration)
    X_fit, y_fit = prepare_eval_frame(calibration_fit, aggregates)
    X_selection, y_selection = prepare_eval_frame(calibration_selection, aggregates)
    X_full, y_full = prepare_eval_frame(calibration, aggregates)

    raw_fit = model.pipeline.predict_proba(X_fit)[:, 1]
    raw_selection = model.pipeline.predict_proba(X_selection)[:, 1]
    raw_full = model.pipeline.predict_proba(X_full)[:, 1]
    calibrator, calibration_report = select_calibrator_on_holdout(
        raw_fit,
        y_fit,
        raw_selection,
        y_selection,
        refit_raw_probabilities=raw_full,
        refit_y=y_full,
    )
    calibrated_full = calibrator.transform(raw_full)
    threshold_result = tune_threshold_for_f1(y_full, calibrated_full)
    calibration_report["fit_dates"] = list(_date_range(calibration_fit))
    calibration_report["selection_dates"] = list(_date_range(calibration_selection))
    calibration_report["full_dates"] = list(_date_range(calibration))
    calibration_report["threshold_tuning"] = {
        "metric": "f1_on_complete_calibration_block",
        "threshold": threshold_result.threshold,
        "f1": threshold_result.f1,
        "precision": threshold_result.precision,
        "recall": threshold_result.recall,
    }
    return calibrator, threshold_result, calibration_report


def _load_manifest(path: Path, data_path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "manifest_missing": True,
        "output_path": str(data_path.resolve()),
        "output_sha256": sha256_file(data_path),
    }


def _candidate_diagnostics(models: dict[str, Any], candidate_keys: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key in candidate_keys:
        pipeline = models[key].pipeline
        estimator = pipeline.named_steps.get("model")
        evidence: dict[str, Any] = {
            "estimator_class": estimator.__class__.__name__ if estimator is not None else "unknown",
        }
        for attribute in (
            "architecture",
            "n_epochs_trained_",
            "best_validation_loss_",
            "parameter_count_",
            "category_cardinalities_",
        ):
            if estimator is not None and hasattr(estimator, attribute):
                value = getattr(estimator, attribute)
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                evidence[attribute.rstrip("_")] = value
        wrapped = getattr(estimator, "model_", None)
        if wrapped is not None and hasattr(wrapped, "n_iter_"):
            evidence["iterations_trained"] = int(wrapped.n_iter_)
        output[key] = evidence
    return output


def _save_candidate_benchmark(
    validation_results: dict[str, dict[str, Any]],
    selected_key: str,
    selection_metric: str,
    diagnostics: dict[str, dict[str, Any]] | None = None,
) -> None:
    payload = {
        "generated_by": "python -m scripts.train_model",
        "selection_metric": selection_metric,
        "selected_model_key": selected_key,
        "candidates": {
            key: value["metrics"] for key, value in sorted(validation_results.items())
        },
        "diagnostics": diagnostics or {},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "candidate_benchmark.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Candidate benchmark",
        "",
        "Generated by `python -m scripts.train_model` on the chronological selection block.",
        "",
        f"Selected candidate: **{selected_key}** using `{selection_metric}`.",
        "",
        "| Candidate | ROC-AUC | PR-AUC | F1 | Brier | Lift@10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, result in sorted(validation_results.items()):
        metrics = result["metrics"]
        lines.append(
            f"| `{key}` | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['brier_score']:.4f} | "
            f"{metrics['lift_at_top_10pct']:.3f}× |"
        )
    (REPORTS_DIR / "candidate_benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def _save_calibration_report(
    *,
    model_name: str,
    calibration_summary: dict[str, Any],
    test_results: dict[str, Any],
    test_dates: tuple[str | None, str | None],
    test_rows: int,
) -> None:
    selected = calibration_summary["selected_model"]
    raw = test_results["raw_probability_metrics"]
    calibrated = {
        key: test_results["metrics"][key]
        for key in (
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "mean_predicted_probability",
            "observed_positive_rate",
        )
    }
    payload = {
        "release": APP_VERSION,
        "model": model_name,
        "method": selected["selected_method"],
        "protocol": calibration_summary["protocol"],
        "method_selection": {
            "fit_period": selected["fit_dates"],
            "selection_period": selected["selection_dates"],
            "fit_rows": selected["fit_rows"],
            "selection_rows": selected["selection_rows"],
            "candidate_metrics_on_later_holdout": selected["candidate_metrics_on_holdout"],
        },
        "final_refit": {
            "period": selected["full_dates"],
            "rows": selected["refit_rows"],
        },
        "threshold_tuning": selected["threshold_tuning"],
        "held_out_test": {
            "start": test_dates[0],
            "end": test_dates[1],
            "rows": test_rows,
        },
        "raw": raw,
        "calibrated": calibrated,
        "improvement": {
            "brier_score_reduction": raw["brier_score"] - calibrated["brier_score"],
            "expected_calibration_error_reduction": (
                raw["expected_calibration_error"]
                - calibrated["expected_calibration_error"]
            ),
            "log_loss_reduction": raw["log_loss"] - calibrated["log_loss"],
        },
        "note": (
            "Calibration method is selected on a chronological holdout inside the "
            "calibration block, then refitted on the complete calibration block. "
            "The final test period is evaluated once and never used for fitting."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "calibration_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    candidates = selected["candidate_metrics_on_holdout"]
    lines = [
        "# Calibration report",
        "",
        f"Selected method: **{selected['selected_method']}** using Brier score on a later chronological holdout.",
        "",
        f"Calibration fit block: `{selected['fit_dates'][0]}` → `{selected['fit_dates'][1]}` ({selected['fit_rows']:,} rows).",
        f"Method-selection block: `{selected['selection_dates'][0]}` → `{selected['selection_dates'][1]}` ({selected['selection_rows']:,} rows).",
        f"Final refit block: `{selected['full_dates'][0]}` → `{selected['full_dates'][1]}` ({selected['refit_rows']:,} rows).",
        "",
        "| Method | Brier | Log loss | ECE |",
        "|---|---:|---:|---:|",
    ]
    for method, values in candidates.items():
        if "error" in values:
            lines.append(f"| `{method}` | error | error | error |")
        else:
            lines.append(
                f"| `{method}` | {values['brier_score']:.4f} | "
                f"{values['log_loss']:.4f} | {values['expected_calibration_error']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Untouched test period",
            "",
            f"`{test_dates[0]}` → `{test_dates[1]}` ({test_rows:,} rows)",
            "",
            "| Metric | Raw | Calibrated |",
            "|---|---:|---:|",
            f"| Brier | {raw['brier_score']:.4f} | {calibrated['brier_score']:.4f} |",
            f"| Log loss | {raw['log_loss']:.4f} | {calibrated['log_loss']:.4f} |",
            f"| ECE | {raw['expected_calibration_error']:.4f} | {calibrated['expected_calibration_error']:.4f} |",
            "",
            "The test outcomes were not used for candidate selection, calibration fitting, method selection or threshold tuning.",
        ]
    )
    (REPORTS_DIR / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def _config_fingerprint(config: TrainingConfig) -> str:
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def run_training(config: TrainingConfig) -> dict[str, Any]:
    with MLflowRun(
        "train-flight-delay-risk", tags={"project": "Flight Delay Risk", "version": APP_VERSION}
    ) as run:
        run.log_params(asdict(config))
        schedule_context = load_or_fit_schedule_context_from_parquet(
            config.data, config.schedule_context
        )
        if config.max_rows is not None:
            logger.warning("Using a deterministic across-month sample of %d rows", config.max_rows)
        df = read_release_frame(config.data, config.max_rows)

        partitions: TemporalPartitions = split_model_selection_calibration_test(
            df,
            test_size=config.test_size,
            calibration_size=config.calibration_size,
            selection_size=config.selection_size,
        )

        models, selection_aggregates, _, _ = train_models(
            partitions.model_train,
            include_gradient_boosting=config.include_gradient_boosting,
            ordered_historical_encoding=True,
            smoothing_strength=config.smoothing_strength,
            candidate_profile=config.candidate_profile,
            schedule_context=schedule_context,
        )
        X_selection, y_selection = prepare_eval_frame(
            partitions.selection, selection_aggregates
        )
        candidate_keys = candidate_keys_for_profile(
            config.candidate_profile,
            include_gradient_boosting=config.include_gradient_boosting,
        )
        validation_results: dict[str, dict[str, Any]] = {}
        for key in candidate_keys:
            result = evaluate_model(
                models[key].pipeline,
                models[key].name,
                X_selection,
                y_selection,
                threshold=0.5,
            )
            validation_results[key] = result
            run.log_metrics(result["metrics"], prefix=f"selection_{models[key].name}_")

        selected_key = _choose_best_model(validation_results, config.selection_metric)
        _save_candidate_benchmark(
            validation_results,
            selected_key,
            config.selection_metric,
            diagnostics=_candidate_diagnostics(models, candidate_keys),
        )
        logger.info("Selected %s on chronological selection %s", selected_key, config.selection_metric)

        refit_df = (
            pd.concat([partitions.model_train, partitions.selection], ignore_index=True)
            .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
            .reset_index(drop=True)
        )
        selected_model, selected_aggregates, X_refit, _ = train_candidate(
            refit_df,
            selected_key,
            smoothing_strength=config.smoothing_strength,
            schedule_context=schedule_context,
        )
        if selected_key == "baseline":
            baseline_model = selected_model
            baseline_aggregates = selected_aggregates
        else:
            baseline_model, baseline_aggregates, _, _ = train_candidate(
                refit_df,
                "baseline",
                smoothing_strength=config.smoothing_strength,
                schedule_context=schedule_context,
            )

        selected_calibrator, selected_threshold, selected_calibration = _fit_calibrator_and_threshold(
            selected_model, selected_aggregates, partitions.calibration
        )
        baseline_calibrator, baseline_threshold, baseline_calibration = _fit_calibrator_and_threshold(
            baseline_model, baseline_aggregates, partitions.calibration
        )

        X_test, y_test = prepare_eval_frame(partitions.test, selected_aggregates)
        X_test_baseline, y_test_baseline = prepare_eval_frame(
            partitions.test, baseline_aggregates
        )
        selected_test_results = evaluate_model(
            selected_model.pipeline,
            selected_model.name,
            X_test,
            y_test,
            threshold=selected_threshold.threshold,
            calibrator=selected_calibrator,
            bootstrap_samples=config.bootstrap_samples,
        )
        baseline_test_results = evaluate_model(
            baseline_model.pipeline,
            baseline_model.name,
            X_test_baseline,
            y_test_baseline,
            threshold=baseline_threshold.threshold,
            calibrator=baseline_calibrator,
            bootstrap_samples=config.bootstrap_samples,
        )

        split_dates = {
            name: list(_date_range(frame)) for name, frame in partitions.as_dict().items()
        }
        split_rows = {name: len(frame) for name, frame in partitions.as_dict().items()}
        selection_summary = {
            "selected_model_key": selected_key,
            "selected_model_name": selected_model.name,
            "selection_metric": config.selection_metric,
            "selection_metrics": {
                key: validation_results[key]["metrics"] for key in sorted(validation_results)
            },
            "split_rows": split_rows,
            "split_dates": split_dates,
            "refit_rows": len(refit_df),
            "decision_threshold": selected_threshold.threshold,
            "baseline_decision_threshold": baseline_threshold.threshold,
        }
        calibration_summary = {
            "protocol": "chronological_holdout_method_selection_then_full_calibration_refit",
            "selected_model": selected_calibration,
            "baseline": baseline_calibration,
        }
        save_reports(
            selected_test_results,
            baseline_test_results,
            out_dir=REPORTS_DIR,
            selection_summary=selection_summary,
            calibration_summary=calibration_summary,
        )
        _save_calibration_report(
            model_name=selected_model.name,
            calibration_summary=calibration_summary,
            test_results=selected_test_results,
            test_dates=_date_range(partitions.test),
            test_rows=len(partitions.test),
        )

        selected_test_raw = selected_model.pipeline.predict_proba(X_test)[:, 1]
        selected_test_proba = selected_calibrator.transform(selected_test_raw)
        error_analysis = build_error_analysis(
            X_test,
            y_test,
            selected_test_proba,
            selected_threshold.threshold,
        )
        save_error_analysis(error_analysis, out_dir=REPORTS_DIR)
        save_drift_reference(build_drift_reference(X_refit))

        data_manifest = _load_manifest(config.data_manifest, config.data)
        framework_by_candidate = {
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "mlp_embeddings": "pytorch",
            "ft_transformer": "pytorch",
        }
        metadata = build_metadata(
            model_name=selected_model.name,
            n_train=len(refit_df),
            n_test=len(partitions.test),
            extra={
                "version": APP_VERSION,
                "release_name": RELEASE_NAME,
                "artifact_schema_version": "5",
                "training_protocol": "train_selection_refit_calibration_test",
                "feature_set": "calendar_congestion_recency_support_v1",
                "feature_families": {name: len(columns) for name, columns in FEATURE_FAMILIES.items()},
                "schedule_context": {
                    "scope": "complete_target_free_published_timetable",
                    "rows": schedule_context.fitted_rows,
                    "start": schedule_context.date_start,
                    "end": schedule_context.date_end,
                },
                "recency_windows_days": [28, 90],
                "ewma_half_life_days": 28,
                "historical_encoding": "strictly_prior_flight_date",
                "smoothing_strength": config.smoothing_strength,
                "calibration_method": selected_calibrator.method,
                "calibration_protocol": calibration_summary["protocol"],
                "candidate_models": [models[key].name for key in candidate_keys],
                "selected_model_key": selected_key,
                "selected_framework": framework_by_candidate.get(selected_key, "scikit-learn"),
                "selection_metric": config.selection_metric,
                "decision_threshold": selected_threshold.threshold,
                "split_rows": split_rows,
                "split_dates": split_dates,
                "data_manifest_path": str(config.data_manifest),
                "data_sha256": data_manifest.get("output_sha256"),
                "data_preparation_config_sha256": data_manifest.get("preparation_config_sha256"),
                "training_config_sha256": _config_fingerprint(config),
                "monitoring": "prediction logging + PSI drift reference",
                "explanation_method": (
                    "exact_linear_log_odds_contributions"
                    if selected_key == "baseline"
                    else (
                        "tree_path_probability_decomposition_rescaled_to_log_odds"
                        if selected_key in {"random_forest", "extra_trees"}
                        else "raw_feature_neutralisation_log_odds_sensitivity"
                    )
                ),
                "neural_inner_validation": (
                    "chronological_training_tail_with_early_stopping"
                    if selected_key in {"mlp_embeddings", "ft_transformer"}
                    else None
                ),
            },
        )
        artifact = FlightRiskArtifact(
            pipeline=selected_model.pipeline,
            historical_aggregates=selected_aggregates,
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
            decision_threshold=selected_threshold.threshold,
            probability_calibrator=selected_calibrator,
        )
        artifact_path = artifact.save(config.output)
        run.log_artifact(artifact_path)
        for report_name in (
            "metrics.json",
            "candidate_benchmark.json",
            "candidate_benchmark.md",
            "feature_importance.csv",
            "error_analysis.json",
            "error_analysis.md",
        ):
            run.log_artifact(REPORTS_DIR / report_name)

        return {
            "artifact_path": str(artifact_path),
            "selected_model_key": selected_key,
            "selected_model_name": selected_model.name,
            "test_metrics": selected_test_results["metrics"],
            "baseline_test_metrics": baseline_test_results["metrics"],
            "split_rows": split_rows,
            "split_dates": split_dates,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flight Delay Risk with strict temporal evaluation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--calibration-size", type=float, default=0.15)
    parser.add_argument("--selection-size", type=float, default=0.20)
    parser.add_argument("--selection-metric", choices=["roc_auc", "pr_auc", "f1"], default="pr_auc")
    parser.add_argument("--include-gradient-boosting", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument("--candidate-profile", choices=list(PROFILE_CHOICES), default="full")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))
    result = run_training(config)
    logger.info("Training complete: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
