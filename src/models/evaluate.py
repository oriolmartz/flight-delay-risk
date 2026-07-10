"""Model evaluation: ranking, classification, calibration and feature evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REPORTS_DIR
from src.models.calibration import probability_metrics
from src.models.uncertainty import compute_metric_confidence_intervals
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ranking_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba).astype(float)
    n = len(y)
    baseline_rate = float(np.mean(y)) if n else 0.0
    order = np.argsort(-p)
    out: dict[str, Any] = {"baseline_positive_rate": baseline_rate}
    for frac in fractions:
        k = max(1, int(round(n * frac))) if n else 0
        top = y[order[:k]] if k else np.array([])
        precision = float(np.mean(top)) if k else 0.0
        recall = float(np.sum(top) / max(np.sum(y), 1)) if k else 0.0
        lift = float(precision / baseline_rate) if baseline_rate > 0 else 0.0
        suffix = f"top_{int(frac * 100)}pct"
        out[f"precision_at_{suffix}"] = precision
        out[f"recall_at_{suffix}"] = recall
        out[f"lift_at_{suffix}"] = lift
        out[f"n_at_{suffix}"] = int(k)
    return out


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "n_samples": int(len(y_true)),
    }
    metrics.update(ranking_metrics(y_true, y_proba))
    metrics.update(probability_metrics(y_true, y_proba))
    return metrics


def get_feature_names(pipeline) -> list[str]:
    preprocessing = pipeline.named_steps["preprocessing"]
    names: list[str] = []
    if "cat" in preprocessing.named_transformers_:
        cat_encoder = preprocessing.named_transformers_["cat"]
        names.extend(list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)))
    if "num" in preprocessing.named_transformers_:
        names.extend(list(NUMERIC_FEATURES))
    return names


def get_feature_importance(pipeline, model_name: str) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        logger.warning("Model %s has no feature_importances_ or coef_; skipping.", model_name)
        return None
    if len(importances) != len(feature_names):
        logger.warning(
            "Feature importance length mismatch (%d importances vs %d names); skipping.",
            len(importances),
            len(feature_names),
        )
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    *,
    model_name: str,
    threshold: float = 0.5,
    pipeline=None,
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    """Evaluate explicit probability outputs, calibrated or raw."""
    probabilities = np.asarray(y_proba, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, probabilities)
    confidence_intervals = None
    if bootstrap_samples and bootstrap_samples > 0:
        confidence_intervals = compute_metric_confidence_intervals(
            y_true, probabilities, threshold, n_bootstrap=bootstrap_samples
        )
    report_text = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    frac_pos, mean_pred = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="quantile"
    )
    calibration = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_probability": mean_pred.tolist(),
        "strategy": "quantile",
    }
    importance_df = get_feature_importance(pipeline, model_name) if pipeline is not None else None
    return {
        "model_name": model_name,
        "metrics": metrics,
        "classification_report": report_text,
        "confusion_matrix": cm.tolist(),
        "calibration": calibration,
        "feature_importance": importance_df,
        "confidence_intervals": confidence_intervals,
    }


def evaluate_model(
    pipeline,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    *,
    calibrator=None,
    bootstrap_samples: int = 0,
) -> dict[str, Any]:
    raw_proba = pipeline.predict_proba(X_test)[:, 1]
    y_proba = calibrator.transform(raw_proba) if calibrator is not None else raw_proba
    result = evaluate_probabilities(
        y_test,
        y_proba,
        model_name=model_name,
        threshold=threshold,
        pipeline=pipeline,
        bootstrap_samples=bootstrap_samples,
    )
    result["raw_probability_metrics"] = probability_metrics(y_test, raw_proba)
    return result


def save_reports(
    main_results: dict,
    baseline_results: dict,
    out_dir: Path = REPORTS_DIR,
    selection_summary: dict | None = None,
    calibration_summary: dict | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_metrics = {
        "main_model": {
            "model_name": main_results["model_name"],
            "metrics": main_results["metrics"],
            "raw_probability_metrics": main_results.get("raw_probability_metrics"),
            "calibration": main_results["calibration"],
            "confidence_intervals": main_results.get("confidence_intervals"),
        },
        "baseline_model": {
            "model_name": baseline_results["model_name"],
            "metrics": baseline_results["metrics"],
            "raw_probability_metrics": baseline_results.get("raw_probability_metrics"),
            "calibration": baseline_results["calibration"],
            "confidence_intervals": baseline_results.get("confidence_intervals"),
        },
        "comparison": {
            "roc_auc_improvement": main_results["metrics"]["roc_auc"]
            - baseline_results["metrics"]["roc_auc"],
            "pr_auc_improvement": main_results["metrics"]["pr_auc"]
            - baseline_results["metrics"]["pr_auc"],
            "brier_improvement": baseline_results["metrics"]["brier_score"]
            - main_results["metrics"]["brier_score"],
        },
    }
    if selection_summary is not None:
        combined_metrics["selection"] = selection_summary
    if calibration_summary is not None:
        combined_metrics["calibration_selection"] = calibration_summary
    (out_dir / "metrics.json").write_text(
        json.dumps(combined_metrics, indent=2), encoding="utf-8"
    )

    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write("=== MAIN MODEL (%s) ===\n" % main_results["model_name"])
        handle.write(main_results["classification_report"])
        handle.write("\n\n=== BASELINE MODEL (%s) ===\n" % baseline_results["model_name"])
        handle.write(baseline_results["classification_report"])

    pd.DataFrame(
        main_results["confusion_matrix"],
        index=["Actual_OnTime", "Actual_Delayed"],
        columns=["Predicted_OnTime", "Predicted_Delayed"],
    ).to_csv(out_dir / "confusion_matrix.csv")

    if main_results["feature_importance"] is not None:
        main_results["feature_importance"].to_csv(
            out_dir / "feature_importance.csv", index=False
        )
    logger.info("Saved evaluation reports to %s", out_dir)
