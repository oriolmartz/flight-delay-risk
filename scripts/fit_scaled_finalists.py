"""Isolated fitting stage for the v1.4 scaled finalist release.

Tree fitting uses native/joblib worker pools. Running it in a short-lived process
prevents those pools from interfering with calibration and API smoke checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from scripts.build_layer5_release import _build_finalist_pipelines, _compact_frame, _dates
from scripts.build_schedule_context import load_or_fit_schedule_context_from_parquet
from src.config import DEFAULT_PROCESSED_PATH, RANDOM_SEED, SCHEDULE_CONTEXT_PATH
from src.data.release_sampling import read_release_frame
from src.data.temporal import split_model_selection_calibration_test
from src.models.train import prepare_training_frame


def fit_stage(
    *,
    data_path: Path,
    schedule_context_path: Path,
    output_path: Path,
    max_rows: int,
    drift_sample_rows: int,
) -> dict:
    started = perf_counter()
    context = load_or_fit_schedule_context_from_parquet(data_path, schedule_context_path)
    frame = read_release_frame(data_path, max_rows)
    partitions = split_model_selection_calibration_test(frame)
    refit = (
        pd.concat([partitions.model_train, partitions.selection], ignore_index=True)
        .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
        .reset_index(drop=True)
    )
    split_rows = {
        "model_train": len(partitions.model_train),
        "selection_inherited_into_refit": len(partitions.selection),
        "refit": len(refit),
        "calibration": len(partitions.calibration),
        "test": len(partitions.test),
    }
    split_dates = {
        "model_train": _dates(partitions.model_train),
        "selection_inherited_into_refit": _dates(partitions.selection),
        "refit": _dates(refit),
        "calibration": _dates(partitions.calibration),
        "test": _dates(partitions.test),
    }

    feature_started = perf_counter()
    X_refit, y_refit, aggregates = prepare_training_frame(
        refit,
        ordered_historical_encoding=True,
        schedule_context=context,
    )
    X_refit = _compact_frame(X_refit)
    feature_seconds = perf_counter() - feature_started
    drift_reference = X_refit.sample(
        n=min(drift_sample_rows, len(X_refit)), random_state=RANDOM_SEED
    ).copy()

    pipelines = _build_finalist_pipelines()
    fit_seconds: dict[str, float] = {}
    fit_order = ("baseline", "extra_trees")
    for name in fit_order:
        pipeline = pipelines[name]
        candidate_started = perf_counter()
        print(f"[isolated-fit] fitting {name} on {len(X_refit):,} rows", flush=True)
        if name == "extra_trees":
            with joblib.parallel_backend("threading", n_jobs=4):
                pipeline.fit(X_refit, y_refit)
        else:
            pipeline.fit(X_refit, y_refit)
        fit_seconds[name] = perf_counter() - candidate_started
        if name == "extra_trees":
            pipeline.named_steps["model"].n_jobs = 1
        print(f"[isolated-fit] {name} fitted in {fit_seconds[name]:.1f}s", flush=True)

    payload = {
        "pipelines": pipelines,
        "historical_aggregates": aggregates.to_dict(),
        "drift_reference": drift_reference,
        "sample_rows": len(frame),
        "split_rows": split_rows,
        "split_dates": split_dates,
        "refit_prevalence": float(refit["ArrDel15"].mean()),
        "feature_engineering_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "fit_stage_seconds": perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path, compress=3)
    result = {
        "output": str(output_path),
        "sample_rows": len(frame),
        "refit_rows": len(refit),
        "fit_seconds": fit_seconds,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--schedule-context", type=Path, default=SCHEDULE_CONTEXT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--drift-sample-rows", type=int, default=50_000)
    args = parser.parse_args()
    fit_stage(
        data_path=args.data,
        schedule_context_path=args.schedule_context,
        output_path=args.output,
        max_rows=args.max_rows,
        drift_sample_rows=args.drift_sample_rows,
    )
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
