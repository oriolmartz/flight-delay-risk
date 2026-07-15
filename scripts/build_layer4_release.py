"""Build FlightRisk v1.3.0 robustness and operational-decision release.

The v1.2 model family winner is frozen. This script refits that family under the
same temporal boundary, selects calibration on a later holdout, derives an
operational top-k policy without using final-test labels, and produces clustered
uncertainty and drift evidence on the untouched final period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import (
    CATEGORICAL_FEATURES,
    DATA_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROCESSED_PATH,
    FEATURE_FAMILIES,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
)
from src.data.manifest import sha256_file
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import split_model_selection_calibration_test
from src.models.calibration import ProbabilityCalibrator, probability_metrics
from src.models.decision_policy import (
    PolicyCosts,
    build_policy_frontier,
    evaluate_capacity_policy,
    threshold_for_capacity,
    tune_cost_sensitive_threshold,
)
from src.models.evaluate import evaluate_model
from src.models.registry import FlightRiskArtifact, build_metadata
from src.models.train import prepare_eval_frame, train_candidate
from src.models.uncertainty import (
    compute_block_confidence_intervals,
    paired_block_bootstrap_difference_ci,
)
from src.monitoring.robustness import build_feature_drift_report, rolling_performance_report
from src.version import APP_VERSION, RELEASE_NAME


def _dates(frame: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return [str(dates.min().date()), str(dates.max().date())]


def _fit_calibration_and_policy(
    model,
    aggregates,
    calibration: pd.DataFrame,
    *,
    capacity_fraction: float,
    costs: PolicyCosts,
) -> tuple[ProbabilityCalibrator, dict[str, Any]]:
    fit_frame, selection_frame = time_aware_split(calibration, test_size=0.5)
    X_fit, y_fit = prepare_eval_frame(fit_frame, aggregates)
    X_selection, y_selection = prepare_eval_frame(selection_frame, aggregates)
    X_full, y_full = prepare_eval_frame(calibration, aggregates)
    raw_fit = model.pipeline.predict_proba(X_fit)[:, 1]
    raw_selection = model.pipeline.predict_proba(X_selection)[:, 1]
    raw_full = model.pipeline.predict_proba(X_full)[:, 1]

    methods = ("identity", "sigmoid", "isotonic")
    holdout_calibrators: dict[str, ProbabilityCalibrator] = {}
    candidate_metrics: dict[str, Any] = {}
    for method in methods:
        candidate = ProbabilityCalibrator(method=method).fit(raw_fit, y_fit)
        holdout_calibrators[method] = candidate
        candidate_metrics[method] = probability_metrics(
            y_selection, candidate.transform(raw_selection)
        )
    selected_method = min(methods, key=lambda name: candidate_metrics[name]["brier_score"])
    holdout_calibrator = holdout_calibrators[selected_method]
    holdout_probability = holdout_calibrator.transform(raw_selection)

    frontier = build_policy_frontier(
        y_selection, holdout_probability, costs=costs, tie_breaker=raw_selection
    )
    capacity_evidence = evaluate_capacity_policy(
        y_selection, holdout_probability, capacity_fraction, costs=costs, tie_breaker=raw_selection
    )
    cost_sensitive_evidence = tune_cost_sensitive_threshold(
        y_selection, holdout_probability, costs=costs, max_selected_fraction=0.30
    )

    # Refit only the already selected calibration family. The capacity cut-off is
    # a probability quantile and therefore does not require labels on the refit block.
    final_calibrator = ProbabilityCalibrator(method=selected_method).fit(raw_full, y_full)
    full_probability = final_calibrator.transform(raw_full)
    final_capacity_cutoff = threshold_for_capacity(full_probability, capacity_fraction)

    return final_calibrator, {
        "selected_method": selected_method,
        "candidate_metrics_on_later_holdout": candidate_metrics,
        "fit_dates": _dates(fit_frame),
        "selection_dates": _dates(selection_frame),
        "full_dates": _dates(calibration),
        "fit_rows": len(fit_frame),
        "selection_rows": len(selection_frame),
        "full_rows": len(calibration),
        "capacity_fraction": capacity_fraction,
        "probability_cutoff": final_capacity_cutoff,
        "capacity_policy_on_holdout": capacity_evidence.to_dict(),
        "cost_sensitive_policy_on_holdout": cost_sensitive_evidence.to_dict(),
        "policy_frontier_on_holdout": frontier,
        "costs": costs.__dict__,
        "guardrail": (
            "Calibration family and policy evidence use the later calibration holdout. "
            "The final capacity cutoff is a label-free quantile after refitting the chosen calibrator."
        ),
    }


def _paired_cis(
    y: np.ndarray,
    selected_probability: np.ndarray,
    baseline_probability: np.ndarray,
    dates: pd.Series,
    *,
    samples: int,
) -> dict[str, Any]:
    functions = {
        "pr_auc": lambda a, p: float(average_precision_score(a, p)),
        "roc_auc": lambda a, p: float(roc_auc_score(a, p)),
        "brier_score": lambda a, p: float(brier_score_loss(a, p)),
        "lift_at_top_10pct": lambda a, p: _lift_at_ten(a, p),
    }
    return {
        name: paired_block_bootstrap_difference_ci(
            y,
            selected_probability,
            baseline_probability,
            dates,
            fn,
            n_bootstrap=samples,
            random_state=71 + offset,
            frequency="week",
        )
        for offset, (name, fn) in enumerate(functions.items())
    }


def _lift_at_ten(y: np.ndarray, p: np.ndarray) -> float:
    prevalence = float(np.mean(y))
    if prevalence <= 0:
        return 0.0
    k = max(1, int(round(len(y) * 0.10)))
    idx = np.argsort(-p, kind="stable")[:k]
    return float(np.mean(y[idx]) / prevalence)


def _write_markdown(
    operational: dict[str, Any], robustness: dict[str, Any], drift: dict[str, Any]
) -> None:
    policy = operational["selected_policy"]
    test = operational["untouched_test"]
    lines = [
        "# Operational policy report",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        "FlightRisk separates calibrated risk estimation from the finite capacity available for human review.",
        "",
        "## Selected policy",
        "",
        f"- Policy: `{policy['policy_name']}`",
        f"- Review capacity: **{policy['capacity_fraction']:.0%}** of a schedule",
        f"- Single-flight reference cutoff: **{policy['probability_cutoff']:.4f}**",
        f"- Calibration method: `{operational['calibration']['selected_method']}`",
        "",
        "## Untouched test evidence",
        "",
        f"- Flights reviewed: {test['selected_count']:,} / {test['rows']:,}",
        f"- Precision: {test['precision']:.4f}",
        f"- Recall: {test['recall']:.4f}",
        f"- Lift: {test['lift']:.3f}×",
        f"- Net utility under declared costs: {test['net_utility']:.2f}",
        "",
        "The test outcomes were not used to select model family, calibration family, capacity fraction or probability cutoff.",
    ]
    (REPORTS_DIR / "operational_policy.md").write_text("\n".join(lines), encoding="utf-8")

    ci = robustness["weekly_block_confidence_intervals"]
    lines = [
        "# Robustness audit",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        "Intervals resample complete operational weeks, not individual flights.",
        "",
        "| Metric | Point | 95% lower | 95% upper |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in ci.items():
        point = robustness["point_metrics"][metric]
        lines.append(
            f"| `{metric}` | {point:.4f} | {values['lower']:.4f} | {values['upper']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparison against logistic baseline",
            "",
            "Positive differences favour the selected model except Brier score, where negative is better.",
            "",
            "| Metric | Mean difference | 95% interval | Excludes zero |",
            "|---|---:|---:|:---:|",
        ]
    )
    for metric, values in robustness["paired_difference_vs_baseline"].items():
        lines.append(
            f"| `{metric}` | {values['mean']:+.4f} | [{values['lower']:+.4f}, {values['upper']:+.4f}] | "
            f"{'yes' if values['excludes_zero'] else 'no'} |"
        )
    (REPORTS_DIR / "robustness_audit.md").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# Temporal drift report",
        "",
        f"Release: **v{APP_VERSION}**",
        "",
        f"Overall feature-drift status: **{drift['feature_drift']['status']}**",
        "",
        drift["expected_temporal_shift_note"],
        "",
        "## Drift by feature family",
        "",
        "| Family | High | Moderate | Low | Maximum value |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, values in drift["feature_family_drift"].items():
        lines.append(
            f"| `{family}` | {values['high']} | {values['moderate']} | {values['low']} | "
            f"{values['max_drift_value']:.4f} |"
        )
    lines.extend([
        "",
        "## Monthly final-test performance",
        "",
        "| Period | Rows | Prevalence | Mean p | PR-AUC | Brier | ECE | Lift@10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in drift["monthly_performance"]:
        pr_auc = "n/a" if row["pr_auc"] is None else f"{row['pr_auc']:.4f}"
        lines.append(
            f"| {row['start']} → {row['end']} | {row['rows']:,} | {row['prevalence']:.4f} | "
            f"{row['mean_probability']:.4f} | {pr_auc} | {row['brier_score']:.4f} | "
            f"{row['expected_calibration_error']:.4f} | {row['lift_at_top_10pct']:.3f}× |"
        )
    (REPORTS_DIR / "drift_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def build_release(
    *,
    data_path: Path,
    schedule_context_path: Path,
    output_path: Path,
    max_rows: int,
    selected_key: str,
    capacity_fraction: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    context = load_or_fit_schedule_context_from_parquet(data_path, schedule_context_path)
    frame = read_release_frame(data_path, max_rows)
    partitions = split_model_selection_calibration_test(frame)
    refit = (
        pd.concat([partitions.model_train, partitions.selection], ignore_index=True)
        .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
        .reset_index(drop=True)
    )
    selected_model, selected_aggregates, X_refit, _ = train_candidate(
        refit, selected_key, schedule_context=context
    )
    baseline_model, baseline_aggregates, _, _ = train_candidate(
        refit, "baseline", schedule_context=context
    )
    costs = PolicyCosts()
    selected_calibrator, selected_policy = _fit_calibration_and_policy(
        selected_model,
        selected_aggregates,
        partitions.calibration,
        capacity_fraction=capacity_fraction,
        costs=costs,
    )
    baseline_calibrator, baseline_policy = _fit_calibration_and_policy(
        baseline_model,
        baseline_aggregates,
        partitions.calibration,
        capacity_fraction=capacity_fraction,
        costs=costs,
    )

    X_test, y_test = prepare_eval_frame(partitions.test, selected_aggregates)
    X_test_baseline, _ = prepare_eval_frame(partitions.test, baseline_aggregates)
    selected_raw = selected_model.pipeline.predict_proba(X_test)[:, 1]
    baseline_raw = baseline_model.pipeline.predict_proba(X_test_baseline)[:, 1]
    selected_probability = selected_calibrator.transform(selected_raw)
    baseline_probability = baseline_calibrator.transform(baseline_raw)
    selected_cutoff = float(selected_policy["probability_cutoff"])
    baseline_cutoff = float(baseline_policy["probability_cutoff"])

    selected_result = evaluate_model(
        selected_model.pipeline,
        selected_model.name,
        X_test,
        y_test,
        threshold=selected_cutoff,
        calibrator=selected_calibrator,
    )
    baseline_result = evaluate_model(
        baseline_model.pipeline,
        baseline_model.name,
        X_test_baseline,
        y_test,
        threshold=baseline_cutoff,
        calibrator=baseline_calibrator,
    )
    test_policy = evaluate_capacity_policy(
        y_test, selected_probability, capacity_fraction, costs=costs, tie_breaker=selected_raw
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
    block_cis = compute_block_confidence_intervals(
        y_test,
        selected_probability,
        partitions.test["FlightDate"],
        n_bootstrap=bootstrap_samples,
        frequency="week",
    )
    paired = _paired_cis(
        y_test.to_numpy(),
        selected_probability,
        baseline_probability,
        partitions.test["FlightDate"],
        samples=bootstrap_samples,
    )

    numeric = [column for column in NUMERIC_FEATURES if column in X_refit and column in X_test]
    categorical = [
        column for column in CATEGORICAL_FEATURES if column in X_refit and column in X_test
    ]
    feature_drift = build_feature_drift_report(
        X_refit,
        X_test,
        numeric_features=numeric,
        categorical_features=categorical,
    )
    family_drift: dict[str, Any] = {}
    for family, columns in FEATURE_FAMILIES.items():
        entries = {
            column: feature_drift["features"][column]
            for column in columns
            if column in feature_drift["features"]
        }
        family_drift[family] = {
            "features_evaluated": len(entries),
            "high": sum(value["level"] == "high" for value in entries.values()),
            "moderate": sum(value["level"] == "moderate" for value in entries.values()),
            "low": sum(value["level"] == "low" for value in entries.values()),
            "max_drift_value": max((value["value"] for value in entries.values()), default=0.0),
        }
    monthly = rolling_performance_report(
        partitions.test["FlightDate"],
        y_test,
        selected_probability,
        ranking_tie_breaker=selected_raw,
    )

    operational = {
        "release": APP_VERSION,
        "model_family_frozen_from": "v1.2.0 candidate selection",
        "selected_model_key": selected_key,
        "selected_model_name": selected_model.name,
        "selected_policy": {
            "policy_name": f"top_{int(round(capacity_fraction * 100))}pct_capacity",
            "capacity_fraction": capacity_fraction,
            "probability_cutoff": selected_cutoff,
            "cutoff_role": "single-flight reference; batch decisions use exact top-k capacity",
        },
        "calibration": selected_policy,
        "cost_sensitive_alternative": selected_policy["cost_sensitive_policy_on_holdout"],
        "untouched_test": test_policy,
        "test_dates": _dates(partitions.test),
        "guardrail": "Final-test labels are used only for reporting policy performance.",
    }
    robustness = {
        "release": APP_VERSION,
        "protocol": "paired_week_block_bootstrap_on_untouched_final_test",
        "bootstrap_samples": bootstrap_samples,
        "point_metrics": point_metrics,
        "weekly_block_confidence_intervals": block_cis,
        "paired_difference_vs_baseline": paired,
        "selected_model_metrics": selected_result["metrics"],
        "baseline_metrics": baseline_result["metrics"],
        "interpretation": (
            "Intervals reflect temporal clustering. A point estimate is not treated as a stable improvement "
            "when the paired confidence interval includes zero."
        ),
    }
    drift = {
        "release": APP_VERSION,
        "reference_dates": _dates(refit),
        "current_dates": _dates(partitions.test),
        "feature_drift": feature_drift,
        "feature_family_drift": family_drift,
        "expected_temporal_shift_note": (
            "Calendar, season, recency and cumulative-support variables are expected to shift "
            "between January-September reference data and October-December evaluation. "
            "Performance monitoring is therefore interpreted alongside feature drift."
        ),
        "monthly_performance": monthly,
        "prevalence_change": float(y_test.mean() - refit["ArrDel15"].mean()),
        "probability_prevalence_gap": float(selected_probability.mean() - y_test.mean()),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("operational_policy", operational),
        ("robustness_audit", robustness),
        ("drift_analysis", drift),
    ):
        (REPORTS_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    _write_markdown(operational, robustness, drift)

    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = build_metadata(
        selected_model.name,
        len(refit),
        len(partitions.test),
        extra={
            "version": APP_VERSION,
            "release_name": RELEASE_NAME,
            "artifact_schema_version": "6",
            "training_protocol": "frozen_model_family_refit_calibration_holdout_policy_test",
            "selected_model_key": selected_key,
            "feature_set": "calendar_congestion_recency_support_v1",
            "feature_families": {name: len(columns) for name, columns in FEATURE_FAMILIES.items()},
            "calibration_method": selected_calibrator.method,
            "operational_policy_name": operational["selected_policy"]["policy_name"],
            "operational_capacity_fraction": capacity_fraction,
            "decision_threshold": selected_cutoff,
            "uncertainty_protocol": "paired_week_block_bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "drift_protocol": "numeric_psi_categorical_js_monthly_performance",
            "data_sha256": manifest.get("output_sha256", sha256_file(data_path)),
            "data_manifest_path": str(DATA_MANIFEST_PATH),
            "schedule_context_rows": context.fitted_rows,
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
            "feature_stability_report": "reports/feature_stability.json",
            "policy_backtest_report": "reports/policy_backtest.json",
            "candidate_models": [
                "logistic_baseline",
                "elastic_net",
                "extra_trees",
                "hist_gradient_boosting",
                "xgboost",
                "lightgbm",
                "catboost",
                "embedding_mlp",
                "ft_transformer",
            ],
            "monitoring": "PSI + Jensen-Shannon + monthly outcome metrics",
        },
    )
    artifact = FlightRiskArtifact(
        pipeline=selected_model.pipeline,
        historical_aggregates=selected_aggregates,
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
    artifact.save(output_path)

    metrics_payload = {
        "release": APP_VERSION,
        "main_model": {
            "model_name": selected_model.name,
            "metrics": selected_result["metrics"],
            "confidence_intervals": block_cis,
        },
        "baseline_model": {
            "model_name": baseline_model.name,
            "metrics": baseline_result["metrics"],
        },
        "operational_policy": operational,
        "paired_comparison": paired,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    return {
        "release": APP_VERSION,
        "artifact": str(output_path),
        "selected_model": selected_model.name,
        "policy": operational["selected_policy"],
        "test_metrics": selected_result["metrics"],
        "drift_status": feature_drift["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--selected-key", default="extra_trees")
    parser.add_argument("--capacity-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    result = build_release(
        data_path=args.data,
        schedule_context_path=args.schedule_context,
        output_path=args.output,
        max_rows=args.max_rows,
        selected_key=args.selected_key,
        capacity_fraction=args.capacity_fraction,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
