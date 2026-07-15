"""Build FlightRisk v1.4.0 scaled-refit and deployment-readiness release.

The model family and 10% review-capacity policy are frozen from prior releases.
This layer increases the training/evaluation sample, keeps the final dates
untouched, and packages runtime evidence required for a public deployment.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    FEATURE_FAMILIES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
)
from src.data.manifest import sha256_file
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import split_model_selection_calibration_test
from src.features.historical_aggregates import HistoricalAggregates
from src.models.calibration import (
    ProbabilityCalibrator,
    select_calibrator_on_holdout,
)
from src.models.decision_policy import (
    PolicyCosts,
    build_policy_frontier,
    evaluate_capacity_policy,
    threshold_for_capacity,
    tune_cost_sensitive_threshold,
)
from src.models.evaluate import evaluate_model
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.train import prepare_eval_frame
from src.models.uncertainty import (
    compute_block_confidence_intervals,
    paired_block_bootstrap_difference_ci,
)
from src.monitoring.robustness import build_feature_drift_report, rolling_performance_report
from src.version import APP_VERSION, RELEASE_NAME


def _compact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce peak memory without repeated whole-frame block copies."""
    out = frame.copy()
    numeric_columns = [column for column in NUMERIC_FEATURES if column in out.columns]
    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in out.columns]
    if numeric_columns:
        out[numeric_columns] = out[numeric_columns].astype(np.float32, copy=False)
    for column in categorical_columns:
        out[column] = out[column].astype("category")
    return out


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _dates(frame: pd.DataFrame) -> list[str]:
    values = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return [str(values.min().date()), str(values.max().date())]


def _linear_preprocessing() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "float32",
                FunctionTransformer(
                    np.asarray,
                    kw_args={"dtype": np.float32},
                    accept_sparse=True,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES)],
        sparse_threshold=0.0,
    )


def _tree_preprocessing() -> ColumnTransformer:
    numeric = FunctionTransformer(
        np.asarray,
        kw_args={"dtype": np.float32},
        accept_sparse=True,
        feature_names_out="one-to-one",
    )
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", numeric, NUMERIC_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def _build_finalist_pipelines() -> dict[str, Pipeline]:
    return {
        "extra_trees": Pipeline(
            [
                ("preprocessing", _tree_preprocessing()),
                (
                    "model",
                    ExtraTreesClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=160,
                        max_depth=12,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "baseline": Pipeline(
            [
                ("preprocessing", _linear_preprocessing()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        max_iter=100,
                        tol=1e-3,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=8,
                        average=True,
                    ),
                ),
            ]
        ),
    }


def _fit_calibrator(
    raw_fit: np.ndarray,
    y_fit: pd.Series,
    raw_selection: np.ndarray,
    y_selection: pd.Series,
) -> tuple[ProbabilityCalibrator, dict[str, Any], np.ndarray, pd.Series]:
    raw_full = np.concatenate([raw_fit, raw_selection])
    y_full = pd.concat(
        [y_fit.reset_index(drop=True), y_selection.reset_index(drop=True)], ignore_index=True
    )
    calibrator, report = select_calibrator_on_holdout(
        raw_fit,
        y_fit,
        raw_selection,
        y_selection,
        refit_raw_probabilities=raw_full,
        refit_y=y_full,
    )
    return calibrator, report, raw_full, y_full


def _lift_at_ten(y: np.ndarray, probability: np.ndarray) -> float:
    prevalence = float(np.mean(y))
    if prevalence <= 0:
        return 0.0
    count = max(1, int(round(len(y) * 0.10)))
    order = np.argsort(-probability, kind="stable")[:count]
    return float(np.mean(y[order]) / prevalence)


def _paired_intervals(
    y: np.ndarray,
    selected: np.ndarray,
    baseline: np.ndarray,
    dates: pd.Series,
    samples: int,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    metrics = {
        "pr_auc": lambda a, p: float(average_precision_score(a, p)),
        "roc_auc": lambda a, p: float(roc_auc_score(a, p)),
        "brier_score": lambda a, p: float(brier_score_loss(a, p)),
        "lift_at_top_10pct": _lift_at_ten,
    }
    return {
        name: paired_block_bootstrap_difference_ci(
            y,
            selected,
            baseline,
            dates,
            fn,
            n_bootstrap=samples,
            random_state=140 + offset,
            frequency="week",
        )
        for offset, (name, fn) in enumerate(metrics.items())
    }


def _load_previous_metrics() -> dict[str, Any]:
    path = REPORTS_DIR / "history" / "v1.3.0" / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_markdown(
    scale: dict[str, Any],
    operational: dict[str, Any],
    robustness: dict[str, Any],
) -> None:
    current = scale["current_release"]
    previous = scale.get("previous_release", {})
    lines = [
        "# Scaled finalist refit",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        "Extra Trees was frozen before this layer; the scale baseline uses log-loss SGD for tractable sparse optimization. No model-zoo reselection used the final test period.",
        "",
        "## Scale",
        "",
        f"- Sampled flights: **{scale['sample_rows']:,}** of {scale['canonical_rows']:,} canonical rows",
        f"- Finalist refit rows: **{scale['split_rows']['refit']:,}**",
        f"- Calibration rows: **{scale['split_rows']['calibration']:,}**",
        f"- Untouched test rows: **{scale['split_rows']['test']:,}**",
        f"- Scale factor vs v1.3 release sample: **{scale['scale_factor_vs_v1_3']:.1f}×**",
        f"- Peak resident memory: **{scale['peak_rss_mb']:.0f} MB**",
        "",
        "## Untouched test",
        "",
        "| Metric | v1.3 | v1.4 scaled | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in (
        "roc_auc",
        "pr_auc",
        "lift_at_top_10pct",
        "brier_score",
        "expected_calibration_error",
    ):
        old = previous.get(metric)
        new = current[metric]
        old_text = f"{old:.4f}" if old is not None else "n/a"
        delta_text = f"{new - old:+.4f}" if old is not None else "n/a"
        lines.append(f"| `{metric}` | {old_text} | {new:.4f} | {delta_text} |")
    lines.extend(
        [
            "",
            "The comparison uses the same chronological date boundaries but a much larger deterministic sample, so deltas are evidence about the release pipeline rather than a paired claim on identical rows.",
        ]
    )
    (REPORTS_DIR / "scale_refit.md").write_text("\n".join(lines), encoding="utf-8")

    policy = operational["selected_policy"]
    test = operational["untouched_test"]
    lines = [
        "# Operational policy report",
        "",
        f"- Frozen policy: `{policy['policy_name']}`",
        f"- Capacity: {policy['capacity_fraction']:.0%}",
        f"- Calibration method selected inside calibration block: `{operational['calibration']['selected_method']}`",
        f"- Test precision: {test['precision']:.4f}",
        f"- Test recall: {test['recall']:.4f}",
        f"- Test lift: {test['lift']:.3f}×",
        "",
        "The final-test labels are reporting-only.",
    ]
    (REPORTS_DIR / "operational_policy.md").write_text("\n".join(lines), encoding="utf-8")

    ci = robustness["weekly_block_confidence_intervals"]
    lines = [
        "# Scaled robustness audit",
        "",
        f"Bootstrap samples: **{robustness['bootstrap_samples']}** complete-week resamples.",
        "",
        "| Metric | Point | 95% lower | 95% upper |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in ci.items():
        point = robustness["point_metrics"][metric]
        lines.append(
            f"| `{metric}` | {point:.4f} | {values['lower']:.4f} | {values['upper']:.4f} |"
        )
    (REPORTS_DIR / "robustness_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_release(
    *,
    data_path: Path = DEFAULT_PROCESSED_PATH,
    schedule_context_path: Path = SCHEDULE_CONTEXT_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
    max_rows: int = 500_000,
    capacity_fraction: float = 0.10,
    bootstrap_samples: int = 500,
    drift_sample_rows: int = 50_000,
    fit_payload_path: Path | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    previous = _load_previous_metrics()
    external_fit_payload = fit_payload_path is not None
    if fit_payload_path is None:
        fit_payload_path = output_path.parent / ".layer5_scaled_fit_payload.joblib"
        fit_payload_path.parent.mkdir(parents=True, exist_ok=True)
        fit_payload_path.unlink(missing_ok=True)
        fit_env = os.environ.copy()
        fit_env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            }
        )
        fit_command = [
            sys.executable,
            "-m",
            "scripts.fit_scaled_finalists",
            "--data",
            str(data_path),
            "--schedule-context",
            str(schedule_context_path),
            "--output",
            str(fit_payload_path),
            "--max-rows",
            str(max_rows),
            "--drift-sample-rows",
            str(drift_sample_rows),
        ]
        print(f"[scale-refit] isolated finalist fit: {max_rows:,} sampled rows", flush=True)
        subprocess.run(fit_command, check=True, env=fit_env)
    elif not fit_payload_path.exists():
        raise FileNotFoundError(f"Fit payload not found: {fit_payload_path}")
    print(f"[scale-refit] loading fit checkpoint: {fit_payload_path}", flush=True)
    fit_payload = joblib.load(fit_payload_path)
    if not external_fit_payload:
        fit_payload_path.unlink(missing_ok=True)

    pipelines = fit_payload["pipelines"]
    aggregates = HistoricalAggregates.from_dict(fit_payload["historical_aggregates"])
    context = aggregates.schedule_context
    if context is None:
        raise RuntimeError("Scaled fit payload does not contain the schedule context.")
    drift_reference = fit_payload["drift_reference"]
    sample_rows = int(fit_payload["sample_rows"])
    split_rows = fit_payload["split_rows"]
    split_dates = fit_payload["split_dates"]
    refit_prevalence = float(fit_payload["refit_prevalence"])
    feature_seconds = float(fit_payload["feature_engineering_seconds"])
    fit_seconds = fit_payload["fit_seconds"]
    fit_stage_seconds = float(fit_payload["fit_stage_seconds"])

    frame = read_release_frame(data_path, max_rows)
    partitions = split_model_selection_calibration_test(frame)
    calibration_frame = partitions.calibration
    test_frame = partitions.test
    observed_rows = {
        "model_train": len(partitions.model_train),
        "selection_inherited_into_refit": len(partitions.selection),
        "refit": len(partitions.model_train) + len(partitions.selection),
        "calibration": len(calibration_frame),
        "test": len(test_frame),
    }
    if observed_rows != split_rows:
        raise RuntimeError(
            f"Fit/evaluation split mismatch: fit={split_rows}, evaluation={observed_rows}"
        )
    del frame, partitions
    _release_memory()

    calibration_fit, calibration_selection = time_aware_split(calibration_frame, test_size=0.5)
    print("[scale-refit] preparing calibration features", flush=True)
    X_cal_fit, y_cal_fit = prepare_eval_frame(calibration_fit, aggregates)
    X_cal_fit = _compact_frame(X_cal_fit)
    X_cal_selection, y_cal_selection = prepare_eval_frame(calibration_selection, aggregates)
    X_cal_selection = _compact_frame(X_cal_selection)

    raw: dict[str, dict[str, np.ndarray]] = {}
    calibration: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        print(f"[scale-refit] calibration prediction: {name}", flush=True)
        raw_fit = pipeline.predict_proba(X_cal_fit)[:, 1]
        raw_selection = pipeline.predict_proba(X_cal_selection)[:, 1]
        print(f"[scale-refit] calibration fit: {name}", flush=True)
        calibrator, report, raw_full, y_full = _fit_calibrator(
            raw_fit, y_cal_fit, raw_selection, y_cal_selection
        )
        calibration[name] = {"calibrator": calibrator, "report": report, "y_full": y_full}
        raw[name] = {
            "fit": raw_fit,
            "selection": raw_selection,
            "full": raw_full,
            "calibrated_full": calibrator.transform(raw_full),
        }

    print("[scale-refit] calibration candidates fitted", flush=True)
    selected_calibrator: ProbabilityCalibrator = calibration["extra_trees"]["calibrator"]
    baseline_calibrator: ProbabilityCalibrator = calibration["baseline"]["calibrator"]
    selected_holdout_probability = selected_calibrator.transform(raw["extra_trees"]["selection"])
    policy_costs = PolicyCosts()
    capacity_holdout = evaluate_capacity_policy(
        y_cal_selection,
        selected_holdout_probability,
        capacity_fraction,
        costs=policy_costs,
        tie_breaker=raw["extra_trees"]["selection"],
    )
    cost_sensitive = tune_cost_sensitive_threshold(
        y_cal_selection,
        selected_holdout_probability,
        costs=policy_costs,
        max_selected_fraction=0.30,
    )
    policy_frontier = build_policy_frontier(
        y_cal_selection,
        selected_holdout_probability,
        costs=policy_costs,
        tie_breaker=raw["extra_trees"]["selection"],
    )
    selected_cutoff = threshold_for_capacity(
        raw["extra_trees"]["calibrated_full"], capacity_fraction
    )
    baseline_cutoff = threshold_for_capacity(raw["baseline"]["calibrated_full"], capacity_fraction)

    del X_cal_fit, X_cal_selection
    _release_memory()

    print("[scale-refit] preparing untouched test features", flush=True)
    X_test, y_test = prepare_eval_frame(test_frame, aggregates)
    X_test = _compact_frame(X_test)
    selected_raw = pipelines["extra_trees"].predict_proba(X_test)[:, 1]
    baseline_raw = pipelines["baseline"].predict_proba(X_test)[:, 1]
    selected_probability = selected_calibrator.transform(selected_raw)
    baseline_probability = baseline_calibrator.transform(baseline_raw)
    print("[scale-refit] evaluating finalists", flush=True)
    selected_result = evaluate_model(
        pipelines["extra_trees"],
        "extra_trees",
        X_test,
        y_test,
        threshold=selected_cutoff,
        calibrator=selected_calibrator,
    )
    baseline_result = evaluate_model(
        pipelines["baseline"],
        "sgd_numeric_logistic_baseline",
        X_test,
        y_test,
        threshold=baseline_cutoff,
        calibrator=baseline_calibrator,
    )
    test_policy = evaluate_capacity_policy(
        y_test,
        selected_probability,
        capacity_fraction,
        costs=policy_costs,
        tie_breaker=selected_raw,
    ).to_dict()
    test_policy["rows"] = len(y_test)

    point_metrics = {
        name: selected_result["metrics"][name]
        for name in (
            "roc_auc",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
            "lift_at_top_10pct",
        )
    }
    print(f"[scale-refit] weekly bootstrap: {bootstrap_samples} samples", flush=True)
    block_cis = compute_block_confidence_intervals(
        y_test,
        selected_probability,
        test_frame["FlightDate"],
        n_bootstrap=bootstrap_samples,
        frequency="week",
    )
    print("[scale-refit] paired baseline intervals", flush=True)
    paired = _paired_intervals(
        y_test.to_numpy(),
        selected_probability,
        baseline_probability,
        test_frame["FlightDate"],
        bootstrap_samples,
    )

    print("[scale-refit] drift and monthly diagnostics", flush=True)
    drift_current = X_test.sample(
        n=min(drift_sample_rows, len(X_test)), random_state=RANDOM_SEED + 1
    ).copy()
    feature_drift = build_feature_drift_report(
        drift_reference,
        drift_current,
        numeric_features=[column for column in NUMERIC_FEATURES if column in X_test],
        categorical_features=[column for column in CATEGORICAL_FEATURES if column in X_test],
    )
    monthly = rolling_performance_report(
        test_frame["FlightDate"],
        y_test,
        selected_probability,
        ranking_tie_breaker=selected_raw,
    )

    operational = {
        "release": APP_VERSION,
        "model_family_frozen_from": "v1.2.0 candidate selection",
        "policy_frozen_from": "v1.3.0 operational policy",
        "selected_model_key": "extra_trees",
        "selected_model_name": "extra_trees",
        "selected_policy": {
            "policy_name": "top_10pct_capacity",
            "capacity_fraction": capacity_fraction,
            "probability_cutoff": selected_cutoff,
            "cutoff_role": "single-flight reference; batch decisions use exact top-k capacity",
        },
        "calibration": {
            "selected_method": selected_calibrator.method,
            "candidate_metrics_on_later_holdout": calibration["extra_trees"]["report"][
                "candidate_metrics_on_holdout"
            ],
            "fit_dates": _dates(calibration_fit),
            "selection_dates": _dates(calibration_selection),
            "full_dates": _dates(calibration_frame),
            "fit_rows": len(calibration_fit),
            "selection_rows": len(calibration_selection),
            "full_rows": len(calibration_frame),
            "capacity_policy_on_holdout": capacity_holdout.to_dict(),
            "cost_sensitive_policy_on_holdout": cost_sensitive.to_dict(),
            "policy_frontier_on_holdout": policy_frontier,
        },
        "untouched_test": test_policy,
        "test_dates": _dates(test_frame),
        "guardrail": "Final-test labels are reporting-only; model family and capacity were frozen before this release.",
    }
    robustness = {
        "release": APP_VERSION,
        "protocol": "paired_week_block_bootstrap_on_scaled_untouched_test",
        "bootstrap_samples": bootstrap_samples,
        "point_metrics": point_metrics,
        "weekly_block_confidence_intervals": block_cis,
        "paired_difference_vs_baseline": paired,
        "selected_model_metrics": selected_result["metrics"],
        "baseline_metrics": baseline_result["metrics"],
    }
    drift = {
        "release": APP_VERSION,
        "reference_dates": split_dates["refit"],
        "current_dates": _dates(test_frame),
        "sampled_reference_rows": len(drift_reference),
        "sampled_current_rows": len(drift_current),
        "feature_drift": feature_drift,
        "monthly_performance": monthly,
        "prevalence_change": float(y_test.mean() - refit_prevalence),
        "probability_prevalence_gap": float(selected_probability.mean() - y_test.mean()),
    }

    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = build_metadata(
        "extra_trees",
        split_rows["refit"],
        split_rows["test"],
        extra={
            "version": APP_VERSION,
            "release_name": RELEASE_NAME,
            "artifact_schema_version": "7",
            "training_protocol": "frozen_finalists_scaled_refit_calibration_test",
            "selected_model_key": "extra_trees",
            "frozen_model_family_from": "v1.2.0",
            "frozen_policy_from": "v1.3.0",
            "training_sample_rows": sample_rows,
            "canonical_dataset_rows": int(manifest.get("dataset", {}).get("output_rows", 0) or 0),
            "scale_factor_vs_v1_3": sample_rows / 30_000,
            "preprocessing_precision": "extra_trees_float32_compact_ordinal; sgd_numeric_logistic_float32",
            "feature_set": "calendar_congestion_recency_support_v1",
            "feature_families": {name: len(columns) for name, columns in FEATURE_FAMILIES.items()},
            "calibration_method": selected_calibrator.method,
            "operational_policy_name": "top_10pct_capacity",
            "operational_capacity_fraction": capacity_fraction,
            "decision_threshold": selected_cutoff,
            "uncertainty_protocol": "paired_week_block_bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "data_sha256": manifest.get("output_sha256", sha256_file(data_path)),
            "data_manifest_path": str(DATA_MANIFEST_PATH),
            "schedule_context": {
                "scope": "complete_target_free_published_timetable",
                "rows": context.fitted_rows,
                "start": context.date_start,
                "end": context.date_end,
            },
            "historical_encoding": "strictly_prior_flight_date",
            "recency_windows_days": [28, 90],
            "ewma_half_life_days": 28,
            "explanation_method": "tree_path_probability_decomposition_rescaled_to_log_odds",
            "selected_framework": "scikit-learn",
            "calibration_protocol": "later_holdout_then_full_refit",
            "candidate_models": [
                "extra_trees_frozen_finalist",
                "logistic_regression_frozen_baseline",
            ],
            "historical_model_zoo_report": "reports/candidate_benchmark.json",
            "split_rows": split_rows,
            "split_dates": split_dates,
            "deployment_contract": {
                "liveness": "/live",
                "readiness": "/ready",
                "health": "/health",
                "openapi": "/openapi.json",
            },
        },
    )
    artifact = FlightRiskArtifact(
        pipeline=pipelines["extra_trees"],
        historical_aggregates=aggregates,
        metadata=metadata,
        metrics={
            "main_model": selected_result["metrics"],
            "baseline_model": baseline_result["metrics"],
            "operational_policy": operational,
            "robustness": robustness,
            "drift": drift,
        },
        decision_threshold=selected_cutoff,
        probability_calibrator=selected_calibrator,
        operational_policy=operational["selected_policy"],
    )
    print("[scale-refit] saving artifact and reports", flush=True)
    artifact.save(output_path)

    previous_metrics = previous.get("main_model", {}).get("metrics", {})
    scale = {
        "release": APP_VERSION,
        "protocol": "frozen_finalists_scaled_refit_same_chronological_boundaries",
        "canonical_rows": int(manifest.get("dataset", {}).get("output_rows", 0) or 0),
        "sample_rows": sample_rows,
        "scale_factor_vs_v1_3": sample_rows / 30_000,
        "split_rows": split_rows,
        "split_dates": split_dates,
        "feature_engineering_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "isolated_fit_stage_seconds": fit_stage_seconds,
        "total_build_seconds": (perf_counter() - started) + (fit_stage_seconds if external_fit_payload else 0.0),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "artifact_size_bytes": output_path.stat().st_size,
        "preprocessing": "Extra Trees compact ordinal float32; SGD logistic standardized numeric/historical features",
        "current_release": point_metrics,
        "previous_release": {metric: previous_metrics.get(metric) for metric in point_metrics},
        "comparison_guardrail": (
            "The date boundaries match v1.3, but the deterministic sample contains more rows. "
            "Metric deltas are not paired row-level estimates."
        ),
    }

    metrics = {
        "release": APP_VERSION,
        "main_model": {
            "model_name": "extra_trees",
            "metrics": selected_result["metrics"],
            "confidence_intervals": block_cis,
        },
        "baseline_model": {
            "model_name": "sgd_numeric_logistic_baseline",
            "metrics": baseline_result["metrics"],
        },
        "operational_policy": operational,
        "paired_comparison": paired,
        "scale_refit": scale,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("metrics", metrics),
        ("operational_policy", operational),
        ("robustness_audit", robustness),
        ("drift_analysis", drift),
        ("scale_refit", scale),
    ):
        (REPORTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if selected_result.get("feature_importance") is not None:
        selected_result["feature_importance"].to_csv(
            REPORTS_DIR / "feature_importance.csv", index=False
        )
    (REPORTS_DIR / "classification_report.txt").write_text(
        "=== EXTRA TREES ===\n"
        + selected_result["classification_report"]
        + "\n\n=== LOGISTIC BASELINE ===\n"
        + baseline_result["classification_report"],
        encoding="utf-8",
    )
    pd.DataFrame(
        selected_result["confusion_matrix"],
        index=["Actual_OnTime", "Actual_Delayed"],
        columns=["Predicted_OnTime", "Predicted_Delayed"],
    ).to_csv(REPORTS_DIR / "confusion_matrix.csv")
    _write_markdown(scale, operational, robustness)

    del X_test, drift_reference, drift_current
    _release_memory()
    return {
        "release": APP_VERSION,
        "artifact": str(output_path),
        "sample_rows": sample_rows,
        "refit_rows": split_rows["refit"],
        "test_rows": split_rows["test"],
        "calibration_method": selected_calibrator.method,
        "test_metrics": point_metrics,
        "policy_lift": test_policy["lift"],
        "build_seconds": scale["total_build_seconds"],
        "peak_rss_mb": scale["peak_rss_mb"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the scaled FlightRisk v1.4 release.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--capacity-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--drift-sample-rows", type=int, default=50_000)
    parser.add_argument("--fit-payload", type=Path)
    args = parser.parse_args()
    result = build_release(
        data_path=args.data,
        schedule_context_path=args.schedule_context,
        output_path=args.output,
        max_rows=args.max_rows,
        capacity_fraction=args.capacity_fraction,
        bootstrap_samples=args.bootstrap_samples,
        drift_sample_rows=args.drift_sample_rows,
        fit_payload_path=args.fit_payload,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
