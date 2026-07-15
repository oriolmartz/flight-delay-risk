"""Run a chronological drop-one-family feature ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import DEFAULT_PROCESSED_PATH, REPORTS_DIR, SCHEDULE_CONTEXT_PATH
from src.data.release_sampling import read_release_frame
from src.data.temporal import split_model_selection_calibration_test
from src.features.feature_sets import ablation_feature_sets
from src.models.evaluate import evaluate_model
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION


def _markdown(payload: dict) -> str:
    lines = [
        "# FlightRisk feature ablation",
        "",
        f"Release: **v{payload['release']}**",
        "",
        "Each row retrains the same model on the same chronological blocks. Only the named feature family changes.",
        "",
        f"Candidate: `{payload['candidate']}` · selection metric: `pr_auc`",
        "",
        "| Scope | Features | PR-AUC | Δ PR-AUC | Lift@10% | Δ Lift | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| `{row['scope']}` | {row['feature_count']} | {row['pr_auc']:.4f} | "
            f"{row['delta_pr_auc']:+.4f} | {row['lift_at_top_10pct']:.3f}× | "
            f"{row['delta_lift_at_top_10pct']:+.3f}× | {row['roc_auc']:.4f} |"
        )
    lines.extend([
        "", "## Interpretation guardrail", "",
        "A performance fall after removing a family means that family helped on this selection period. "
        "A performance rise means it did not generalise in this run; the negative result is retained rather than reclassified as a win.",
    ])
    return "\n".join(lines)


def run_ablation(data_path: Path, schedule_context_path: Path, max_rows: int, candidate: str) -> dict:
    context = load_or_fit_schedule_context_from_parquet(data_path, schedule_context_path)
    frame = read_release_frame(data_path, max_rows)
    partitions = split_model_selection_calibration_test(frame, test_size=0.20, calibration_size=0.15, selection_size=0.20)
    feature_sets = ablation_feature_sets()
    X_train, y_train, aggregates = prepare_training_frame(
        partitions.model_train, schedule_context=context
    )
    X_selection, y_selection = prepare_eval_frame(partitions.selection, aggregates)
    raw_results = []
    for scope, columns in feature_sets.items():
        pipeline = build_candidate_pipeline(candidate, feature_columns=columns)
        pipeline.fit(X_train[columns], y_train)
        result = evaluate_model(pipeline, candidate, X_selection[columns], y_selection, threshold=0.5)
        metrics = result["metrics"]
        raw_results.append({
            "scope": scope,
            "feature_count": len(columns),
            "features": columns,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "lift_at_top_10pct": metrics["lift_at_top_10pct"],
        })
    baseline = next(row for row in raw_results if row["scope"] == "full")
    for row in raw_results:
        row["delta_pr_auc"] = row["pr_auc"] - baseline["pr_auc"]
        row["delta_lift_at_top_10pct"] = row["lift_at_top_10pct"] - baseline["lift_at_top_10pct"]
    return {
        "release": APP_VERSION,
        "candidate": candidate,
        "selection_metric": "pr_auc",
        "max_rows": max_rows,
        "schedule_context_rows": context.fitted_rows,
        "train_rows": len(partitions.model_train),
        "selection_rows": len(partitions.selection),
        "results": raw_results,
        "guardrail": "Negative ablation results are preserved; evidence is model- and period-specific.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--candidate", default="extra_trees")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "feature_ablation.json")
    args = parser.parse_args()
    payload = run_ablation(args.data, args.schedule_context, args.max_rows, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"release": payload["release"], "results": payload["results"]}, indent=2))


if __name__ == "__main__":
    main()
