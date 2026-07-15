"""Rolling operational-policy backtest for the frozen FlightRisk model family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_layer4_release import _fit_calibration_and_policy
from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import DEFAULT_PROCESSED_PATH, REPORTS_DIR, SCHEDULE_CONTEXT_PATH
from src.data.release_sampling import read_release_frame
from src.data.split import time_aware_split
from src.data.temporal import make_expanding_time_folds
from src.models.decision_policy import PolicyCosts, evaluate_capacity_policy
from src.models.evaluate import evaluate_model
from src.models.train import prepare_eval_frame, train_candidate
from src.version import APP_VERSION


def _date_range(frame: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return [str(dates.min().date()), str(dates.max().date())]


def run_policy_backtest(
    *,
    data_path: Path,
    schedule_context_path: Path,
    max_rows: int,
    selected_key: str,
    capacity_fraction: float,
    n_splits: int,
) -> dict[str, Any]:
    context = load_or_fit_schedule_context_from_parquet(data_path, schedule_context_path)
    frame = read_release_frame(data_path, max_rows)
    folds = make_expanding_time_folds(frame, n_splits=n_splits, min_train_fraction=0.52)
    costs = PolicyCosts()
    rows: list[dict[str, Any]] = []
    for fold_id, (development, future) in enumerate(folds, start=1):
        refit, calibration = time_aware_split(development, test_size=0.15)
        model, aggregates, _, _ = train_candidate(
            refit, selected_key, schedule_context=context
        )
        calibrator, policy = _fit_calibration_and_policy(
            model,
            aggregates,
            calibration,
            capacity_fraction=capacity_fraction,
            costs=costs,
        )
        X_future, y_future = prepare_eval_frame(future, aggregates)
        raw = model.pipeline.predict_proba(X_future)[:, 1]
        probability = calibrator.transform(raw)
        policy_result = evaluate_capacity_policy(
            y_future,
            probability,
            capacity_fraction,
            costs=costs,
            tie_breaker=raw,
        )
        evaluation = evaluate_model(
            model.pipeline,
            model.name,
            X_future,
            y_future,
            threshold=float(policy["probability_cutoff"]),
            calibrator=calibrator,
        )
        rows.append(
            {
                "fold": fold_id,
                "train_dates": _date_range(refit),
                "calibration_dates": _date_range(calibration),
                "test_dates": _date_range(future),
                "train_rows": len(refit),
                "calibration_rows": len(calibration),
                "test_rows": len(future),
                "calibration_method": calibrator.method,
                "probability_cutoff": float(policy["probability_cutoff"]),
                "policy": policy_result.to_dict(),
                "ranking_metrics": {
                    key: evaluation["metrics"][key]
                    for key in (
                        "roc_auc",
                        "pr_auc",
                        "brier_score",
                        "expected_calibration_error",
                        "lift_at_top_10pct",
                    )
                },
            }
        )
    summary: dict[str, Any] = {}
    for metric in ("precision", "recall", "lift", "utility_per_flight"):
        values = np.asarray([row["policy"][metric] for row in rows], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    methods: dict[str, int] = {}
    for row in rows:
        methods[row["calibration_method"]] = methods.get(row["calibration_method"], 0) + 1
    return {
        "release": APP_VERSION,
        "protocol": "fixed_model_family_expanding_window_calibration_then_exact_top_k",
        "selected_model_key": selected_key,
        "capacity_fraction": capacity_fraction,
        "n_splits": len(rows),
        "calibration_method_counts": methods,
        "summary": summary,
        "folds": rows,
        "guardrail": (
            "Every fold refits the frozen model family and calibration using only earlier dates. "
            "The future fold is used once for policy reporting."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Operational policy temporal backtest",
        "",
        f"Release: **v{payload['release']}**",
        "",
        f"Frozen model family: `{payload['selected_model_key']}` · capacity: {payload['capacity_fraction']:.0%}",
        "",
        "| Fold | Test period | Calibration | Precision | Recall | Lift | Utility/flight |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["folds"]:
        policy = row["policy"]
        lines.append(
            f"| {row['fold']} | {row['test_dates'][0]} → {row['test_dates'][1]} | "
            f"`{row['calibration_method']}` | {policy['precision']:.4f} | "
            f"{policy['recall']:.4f} | {policy['lift']:.3f}× | "
            f"{policy['utility_per_flight']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate policy stability",
            "",
            f"- Mean precision: {payload['summary']['precision']['mean']:.4f}",
            f"- Mean recall: {payload['summary']['recall']['mean']:.4f}",
            f"- Mean lift: {payload['summary']['lift']['mean']:.3f}×",
            f"- Lift range: {payload['summary']['lift']['min']:.3f}× → {payload['summary']['lift']['max']:.3f}×",
            "",
            payload["guardrail"],
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--selected-key", default="extra_trees")
    parser.add_argument("--capacity-fraction", type=float, default=0.10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "policy_backtest.json")
    args = parser.parse_args()
    payload = run_policy_backtest(
        data_path=args.data,
        schedule_context_path=args.schedule_context,
        max_rows=args.max_rows,
        selected_key=args.selected_key,
        capacity_fraction=args.capacity_fraction,
        n_splits=args.n_splits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
