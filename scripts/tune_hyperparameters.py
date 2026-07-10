"""Small time-aware hyperparameter search for FlightRisk.

This script exists for technical hardening: it shows how fixed candidate
hyperparameters can be replaced by a reproducible search process. It is
intentionally modest by default so it can run on a laptop.

The search is performed after leakage-safe aggregate features are fit on the
training split only. TimeSeriesSplit is used inside the model pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, RANDOM_SEED, RAW_DATA_DIR, REPORTS_DIR, TARGET_COL
from src.data.clean import clean_flights
from src.data.load_data import load_raw_directory
from src.data.split import split_train_test
from src.features.build_features import add_schedule_features, assert_no_leakage_columns
from src.features.historical_aggregates import HistoricalAggregates
from src.models.train import build_preprocessing_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _candidate_pipeline(model: str) -> tuple[Pipeline, dict]:
    if model == "logistic_regression":
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        params = {
            "model__C": [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.5, 2.0],
            "model__penalty": ["l1", "l2"],
        }
    elif model == "random_forest":
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
            ]
        )
        params = {
            "model__n_estimators": [120, 180, 240],
            "model__max_depth": [8, 10, 14, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        }
    elif model == "extra_trees":
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing_pipeline()),
                ("model", ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
            ]
        )
        params = {
            "model__n_estimators": [160, 240, 320],
            "model__max_depth": [10, 14, 18, None],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        }
    else:
        raise ValueError(f"Unsupported model: {model}")
    return pipeline, params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a modest time-aware hyperparameter search.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--model", choices=["logistic_regression", "random_forest", "extra_trees"], default="logistic_regression")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--scoring", choices=["average_precision", "roc_auc"], default="average_precision")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "hyperparameter_search.json")
    args = parser.parse_args()

    raw_df = load_raw_directory(args.raw_dir)
    clean_df = clean_flights(raw_df)
    if args.max_rows is not None and len(clean_df) > args.max_rows:
        clean_df = clean_df.sample(n=args.max_rows, random_state=RANDOM_SEED).sort_index().reset_index(drop=True)

    train_df, _ = split_train_test(clean_df, test_size=0.2)

    # Fit aggregates on training only and build the exact same leakage-safe
    # feature matrix used by the main pipeline, without training the fixed
    # candidate models first.
    feature_df = add_schedule_features(train_df.copy())
    aggregates = HistoricalAggregates().fit(feature_df)
    feature_df = aggregates.transform(feature_df)
    assert_no_leakage_columns(list(feature_df.columns))
    X_train = feature_df[FEATURE_COLUMNS].copy()
    y_train = feature_df[TARGET_COL].astype(int).copy()

    pipeline, param_distributions = _candidate_pipeline(args.model)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=args.scoring,
        cv=TimeSeriesSplit(n_splits=args.cv_splits),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    output = {
        "model": args.model,
        "scoring": args.scoring,
        "n_iter": args.n_iter,
        "cv_splits": args.cv_splits,
        "max_rows": args.max_rows,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_results_top": [
            {
                "rank": int(row["rank_test_score"]),
                "mean_test_score": float(row["mean_test_score"]),
                "std_test_score": float(row["std_test_score"]),
                "params": row["params"],
            }
            for row in sorted(
                [
                    {
                        "rank_test_score": search.cv_results_["rank_test_score"][i],
                        "mean_test_score": search.cv_results_["mean_test_score"][i],
                        "std_test_score": search.cv_results_["std_test_score"][i],
                        "params": search.cv_results_["params"][i],
                    }
                    for i in range(len(search.cv_results_["params"]))
                ],
                key=lambda r: r["rank_test_score"],
            )[:10]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved hyperparameter search report to %s", args.output)
    print(json.dumps({"best_score": output["best_score"], "best_params": output["best_params"]}, indent=2))


if __name__ == "__main__":
    main()
