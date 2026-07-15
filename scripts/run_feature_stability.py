"""Estimate feature-family stability across pre-test chronological folds.

The script fits a fixed model family on expanding windows and jointly permutes
one feature family at a time on the future fold. It never consults the final
calibration or test outcomes when defining the stable family set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import (
    DEFAULT_PROCESSED_PATH,
    FEATURE_FAMILIES,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
)
from src.data.release_sampling import read_release_frame
from src.data.temporal import make_expanding_time_folds, split_model_selection_calibration_test
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION


def _date_range(frame: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return [str(dates.min().date()), str(dates.max().date())]


def _jointly_permute(frame: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    shuffled = frame.copy()
    permutation = rng.permutation(len(frame))
    for column in columns:
        shuffled[column] = frame[column].iloc[permutation].to_numpy()
    return shuffled


def run_feature_stability(
    *,
    data_path: Path,
    schedule_context_path: Path,
    max_rows: int,
    candidate: str,
    n_splits: int,
    repeats: int,
) -> dict[str, Any]:
    context = load_or_fit_schedule_context_from_parquet(data_path, schedule_context_path)
    frame = read_release_frame(data_path, max_rows)
    partitions = split_model_selection_calibration_test(frame)
    pre_calibration = (
        pd.concat([partitions.model_train, partitions.selection], ignore_index=True)
        .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
        .reset_index(drop=True)
    )
    folds = make_expanding_time_folds(
        pre_calibration, n_splits=n_splits, min_train_fraction=0.52
    )
    fold_results: list[dict[str, Any]] = []
    family_drops: dict[str, list[float]] = {name: [] for name in FEATURE_FAMILIES}

    for fold_id, (train_df, future_df) in enumerate(folds, start=1):
        X_train, y_train, aggregates = prepare_training_frame(
            train_df, schedule_context=context
        )
        X_future, y_future = prepare_eval_frame(future_df, aggregates)
        pipeline = build_candidate_pipeline(candidate)
        pipeline.fit(X_train, y_train)
        base_probability = pipeline.predict_proba(X_future)[:, 1]
        base_pr_auc = float(average_precision_score(y_future, base_probability))
        family_results: dict[str, Any] = {}
        for family_offset, (family, columns) in enumerate(FEATURE_FAMILIES.items()):
            scores: list[float] = []
            for repeat in range(repeats):
                rng = np.random.default_rng(42 + fold_id * 100 + family_offset * 10 + repeat)
                permuted = _jointly_permute(X_future, columns, rng)
                probability = pipeline.predict_proba(permuted)[:, 1]
                scores.append(float(average_precision_score(y_future, probability)))
            permuted_mean = float(np.mean(scores))
            drop = float(base_pr_auc - permuted_mean)
            family_drops[family].append(drop)
            family_results[family] = {
                "baseline_pr_auc": base_pr_auc,
                "permuted_pr_auc_mean": permuted_mean,
                "permuted_pr_auc_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "pr_auc_drop": drop,
                "repeats": repeats,
            }
        fold_results.append(
            {
                "fold": fold_id,
                "train_dates": _date_range(train_df),
                "future_dates": _date_range(future_df),
                "train_rows": len(train_df),
                "future_rows": len(future_df),
                "baseline_pr_auc": base_pr_auc,
                "families": family_results,
            }
        )

    required_positive_folds = max(1, int(np.ceil(len(folds) * 2 / 3)))
    summary: dict[str, Any] = {}
    selected_families = ["core_schedule"]
    for family, values in family_drops.items():
        arr = np.asarray(values, dtype=float)
        positive_folds = int(np.sum(arr > 0))
        stable = family == "core_schedule" or (
            positive_folds >= required_positive_folds and float(np.median(arr)) > 0
        )
        if stable and family not in selected_families:
            selected_families.append(family)
        summary[family] = {
            "mean_pr_auc_drop": float(arr.mean()),
            "median_pr_auc_drop": float(np.median(arr)),
            "min_pr_auc_drop": float(arr.min()),
            "max_pr_auc_drop": float(arr.max()),
            "positive_folds": positive_folds,
            "total_folds": int(len(arr)),
            "stable": stable,
        }

    selected_columns = [
        column
        for family in selected_families
        for column in FEATURE_FAMILIES[family]
    ]
    return {
        "release": APP_VERSION,
        "protocol": "pre_calibration_expanding_folds_joint_family_permutation",
        "candidate": candidate,
        "max_rows": max_rows,
        "n_splits": len(folds),
        "repeats": repeats,
        "final_calibration_and_test_consulted": False,
        "required_positive_folds": required_positive_folds,
        "selected_families": selected_families,
        "selected_feature_count": len(selected_columns),
        "selected_features": selected_columns,
        "summary": summary,
        "folds": fold_results,
        "guardrail": (
            "A family is marked stable only when its joint permutation reduces PR-AUC "
            "in at least two thirds of pre-calibration folds and the median drop is positive."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Feature-family stability audit",
        "",
        f"Release: **v{payload['release']}**",
        "",
        "This audit uses expanding folds inside the pre-calibration period. The final calibration and test outcomes are not consulted.",
        "",
        f"Fixed candidate: `{payload['candidate']}` · permutation repeats: {payload['repeats']}",
        "",
        "| Family | Mean Δ PR-AUC | Median Δ | Positive folds | Stable |",
        "|---|---:|---:|---:|:---:|",
    ]
    for family, row in payload["summary"].items():
        lines.append(
            f"| `{family}` | {row['mean_pr_auc_drop']:+.4f} | "
            f"{row['median_pr_auc_drop']:+.4f} | {row['positive_folds']}/{row['total_folds']} | "
            f"{'yes' if row['stable'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Stable feature policy",
            "",
            f"Selected families: `{', '.join(payload['selected_families'])}`",
            "",
            f"Selected features: **{payload['selected_feature_count']}**",
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
    parser.add_argument("--candidate", default="extra_trees")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "feature_stability.json")
    args = parser.parse_args()
    payload = run_feature_stability(
        data_path=args.data,
        schedule_context_path=args.schedule_context,
        max_rows=args.max_rows,
        candidate=args.candidate,
        n_splits=args.n_splits,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"selected_families": payload["selected_families"]}, indent=2))


if __name__ == "__main__":
    main()
