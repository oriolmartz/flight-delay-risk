"""Leakage-safe temporal hyperparameter search for every FlightRisk family.

Every trial rebuilds ordered historical target features inside every temporal
fold. The search backend can be deterministic random sampling or Optuna; both
use the same fold evidence and never precompute target-derived features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler

from src.config import DEFAULT_PROCESSED_PATH, RANDOM_SEED, REPORTS_DIR
from src.data.io import read_processed_frame
from src.data.temporal import make_expanding_time_folds
from src.models.train import (
    CANDIDATE_NAMES,
    build_candidate_pipeline,
    prepare_eval_frame,
    prepare_training_frame,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parameter_space(model: str) -> dict[str, list[Any]]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "baseline": {
            "model__solver": ["liblinear"],
            "model__C": [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.5, 2.0],
            "model__penalty": ["l1", "l2"],
        },
        "random_forest": {
            "model__n_estimators": [120, 180, 240],
            "model__max_depth": [8, 10, 14, None],
            "model__min_samples_leaf": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None],
        },
        "extra_trees": {
            "model__n_estimators": [160, 240, 320],
            "model__max_depth": [10, 14, 18, None],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        "xgboost": {
            "model__n_estimators": [250, 450, 700],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.025, 0.05, 0.08],
            "model__min_child_weight": [3, 8, 15],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [1.0, 2.0, 5.0],
        },
        "lightgbm": {
            "model__n_estimators": [300, 500, 750],
            "model__learning_rate": [0.025, 0.04, 0.07],
            "model__num_leaves": [24, 48, 72],
            "model__min_child_samples": [30, 60, 100],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_lambda": [0.5, 2.0, 5.0],
        },
        "mlp_embeddings": {
            "model__hidden_dims": [(128, 64), (192, 96, 48), (256, 128, 64)],
            "model__dropout": [0.05, 0.15, 0.25],
            "model__learning_rate": [3e-4, 7e-4, 1e-3],
            "model__weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "model__batch_size": [512, 1024, 2048],
            "model__epochs": [12, 18, 25],
        },
        "ft_transformer": {
            "model__d_token": [32, 48, 64],
            "model__n_heads": [4, 8],
            "model__n_layers": [1, 2, 3],
            "model__dropout": [0.05, 0.15, 0.25],
            "model__learning_rate": [2e-4, 5e-4, 8e-4],
            "model__weight_decay": [1e-5, 1e-4, 1e-3],
            "model__batch_size": [512, 768, 1024],
            "model__epochs": [10, 16, 22],
        },
    }
    if model not in spaces:
        raise ValueError(f"Unsupported model: {model}")
    return spaces[model]


def _score(y_true: pd.Series, probabilities: np.ndarray, scoring: str) -> float:
    if scoring == "average_precision":
        return float(average_precision_score(y_true, probabilities))
    if scoring == "roc_auc":
        return float(roc_auc_score(y_true, probabilities))
    raise ValueError(f"Unsupported scoring: {scoring}")


def evaluate_parameter_set(
    df: pd.DataFrame,
    *,
    model: str,
    params: dict[str, Any],
    cv_splits: int,
    scoring: str,
    smoothing_strength: float,
) -> dict[str, Any]:
    fold_scores: list[float] = []
    fold_evidence: list[dict[str, Any]] = []
    folds = make_expanding_time_folds(df, n_splits=cv_splits, min_train_fraction=0.5)
    for fold_id, (fold_train, fold_validation) in enumerate(folds, start=1):
        X_train, y_train, aggregates = prepare_training_frame(
            fold_train,
            ordered_historical_encoding=True,
            smoothing_strength=smoothing_strength,
        )
        pipeline = build_candidate_pipeline(model, model_params=params)
        pipeline.fit(X_train, y_train)
        X_validation, y_validation = prepare_eval_frame(fold_validation, aggregates)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]
        score = _score(y_validation, probabilities, scoring)
        fold_scores.append(score)
        fold_evidence.append(
            {
                "fold": fold_id,
                "train_start": str(pd.to_datetime(fold_train["FlightDate"]).min().date()),
                "train_end": str(pd.to_datetime(fold_train["FlightDate"]).max().date()),
                "validation_start": str(pd.to_datetime(fold_validation["FlightDate"]).min().date()),
                "validation_end": str(pd.to_datetime(fold_validation["FlightDate"]).max().date()),
                "train_rows": len(fold_train),
                "validation_rows": len(fold_validation),
                "score": score,
            }
        )
    return {
        "params": params,
        "mean_test_score": float(np.mean(fold_scores)),
        "std_test_score": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "folds": fold_evidence,
    }


def _rank_results(
    results: list[dict[str, Any]], *, model: str, scoring: str, cv_splits: int, engine: str
) -> dict[str, Any]:
    ranked = sorted(results, key=lambda item: item["mean_test_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {
        "protocol": "fold_local_ordered_historical_features",
        "search_engine": engine,
        "model": model,
        "scoring": scoring,
        "n_trials": len(ranked),
        "cv_splits": cv_splits,
        "best_score": ranked[0]["mean_test_score"],
        "best_params": ranked[0]["params"],
        "cv_results": ranked,
    }


def run_random_search(
    df: pd.DataFrame,
    *,
    model: str,
    n_iter: int,
    cv_splits: int,
    scoring: str,
    smoothing_strength: float,
) -> dict[str, Any]:
    sampled_params = list(
        ParameterSampler(_parameter_space(model), n_iter=n_iter, random_state=RANDOM_SEED)
    )
    results = []
    for index, params in enumerate(sampled_params, start=1):
        logger.info("Random-search trial %d/%d: %s", index, len(sampled_params), params)
        results.append(
            evaluate_parameter_set(
                df,
                model=model,
                params=params,
                cv_splits=cv_splits,
                scoring=scoring,
                smoothing_strength=smoothing_strength,
            )
        )
    return _rank_results(
        results, model=model, scoring=scoring, cv_splits=cv_splits, engine="random"
    )


def run_optuna_search(
    df: pd.DataFrame,
    *,
    model: str,
    n_iter: int,
    cv_splits: int,
    scoring: str,
    smoothing_strength: float,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna search requires `pip install optuna` or requirements-advanced.txt"
        ) from exc

    space = _parameter_space(model)
    evidence: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = {
            key: trial.suggest_categorical(key.replace("__", "_"), values)
            for key, values in space.items()
        }
        result = evaluate_parameter_set(
            df,
            model=model,
            params=params,
            cv_splits=cv_splits,
            scoring=scoring,
            smoothing_strength=smoothing_strength,
        )
        result["trial_number"] = trial.number
        evidence.append(result)
        return result["mean_test_score"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter)
    return _rank_results(
        evidence, model=model, scoring=scoring, cv_splits=cv_splits, engine="optuna_tpe"
    )


def run_search(
    df: pd.DataFrame,
    *,
    model: str,
    n_iter: int,
    cv_splits: int,
    scoring: str,
    smoothing_strength: float,
    search_engine: str = "random",
) -> dict[str, Any]:
    if search_engine == "random":
        return run_random_search(
            df,
            model=model,
            n_iter=n_iter,
            cv_splits=cv_splits,
            scoring=scoring,
            smoothing_strength=smoothing_strength,
        )
    if search_engine == "optuna":
        return run_optuna_search(
            df,
            model=model,
            n_iter=n_iter,
            cv_splits=cv_splits,
            scoring=scoring,
            smoothing_strength=smoothing_strength,
        )
    raise ValueError("search_engine must be random or optuna")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe temporal hyperparameter search.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--model", choices=list(CANDIDATE_NAMES), default="baseline")
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--scoring", choices=["average_precision", "roc_auc"], default="average_precision")
    parser.add_argument("--search-engine", choices=["random", "optuna"], default="random")
    parser.add_argument("--smoothing-strength", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "hyperparameter_search.json")
    args = parser.parse_args()

    df = read_processed_frame(args.data)
    if args.max_rows is not None and len(df) > args.max_rows:
        df = (
            df.sample(n=args.max_rows, random_state=RANDOM_SEED)
            .sort_values(["FlightDate", "CRSDepTime"], kind="stable")
            .reset_index(drop=True)
        )
    output = run_search(
        df,
        model=args.model,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        scoring=args.scoring,
        smoothing_strength=args.smoothing_strength,
        search_engine=args.search_engine,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Saved leakage-safe hyperparameter search to %s", args.output)


if __name__ == "__main__":
    main()
