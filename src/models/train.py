"""Candidate construction and leakage-safe feature preparation for FlightRisk."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COL,
)
from src.features.build_features import add_schedule_features, assert_no_leakage_columns
from src.features.historical_aggregates import HistoricalAggregates
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline


CANDIDATE_NAMES: dict[str, str] = {
    "baseline": "logistic_regression",
    "random_forest": "random_forest",
    "extra_trees": "extra_trees",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "mlp_embeddings": "mlp_embeddings",
    "ft_transformer": "ft_transformer",
}
MODEL_KEY_ALIASES = {"logistic_regression": "baseline", **{key: key for key in CANDIDATE_NAMES}}

PROFILE_CHOICES: tuple[str, ...] = (
    "baseline",
    "linear",
    "full",
    "trees",
    "boosting",
    "neural",
    "flagship",
    "all",
)

OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "mlp_embeddings": "torch",
    "ft_transformer": "torch",
}



def candidate_is_available(candidate_key: str) -> bool:
    dependency = OPTIONAL_DEPENDENCIES.get(candidate_key)
    return dependency is None or find_spec(dependency) is not None


def unavailable_candidates(candidate_keys: list[str]) -> list[str]:
    return [key for key in candidate_keys if not candidate_is_available(key)]


def candidate_keys_for_profile(
    candidate_profile: str = "full", *, include_gradient_boosting: bool = False
) -> list[str]:
    """Return the intentionally compact public model scope.

    The active zoo compares one interpretable linear baseline, two bagging
    ensembles, two modern boosting libraries and two neural tabular models.
    ``include_gradient_boosting`` is accepted only for backwards-compatible
    CLI calls and no longer adds the retired sklearn HGB candidate.
    """
    if candidate_profile not in PROFILE_CHOICES:
        raise ValueError(f"candidate_profile must be one of: {', '.join(PROFILE_CHOICES)}")

    flagship = [
        "baseline",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "mlp_embeddings",
        "ft_transformer",
    ]
    profiles = {
        "baseline": ["baseline"],
        "linear": ["baseline"],
        "full": ["baseline", "random_forest", "extra_trees"],
        "trees": ["random_forest", "extra_trees"],
        "boosting": ["baseline", "xgboost", "lightgbm"],
        "neural": ["baseline", "mlp_embeddings", "ft_transformer"],
        "flagship": flagship,
        "all": flagship,
    }
    return list(profiles[candidate_profile])


def _feature_partition(feature_columns: list[str] | None = None) -> tuple[list[str], list[str]]:
    columns = list(feature_columns or FEATURE_COLUMNS)
    categorical = [column for column in CATEGORICAL_FEATURES if column in columns]
    numeric = [column for column in columns if column not in categorical]
    return categorical, numeric


def build_preprocessing_pipeline(
    *, dense: bool = False, feature_columns: list[str] | None = None
) -> ColumnTransformer:
    categorical, numeric = _feature_partition(feature_columns)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


def build_compact_tree_preprocessing(
    feature_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Dense compact matrix for histogram trees without one-hot explosion."""
    categorical = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=float,
    )
    categorical_columns, numeric_columns = _feature_partition(feature_columns)
    return ColumnTransformer(
        transformers=[
            ("cat", categorical, categorical_columns),
            ("num", "passthrough", numeric_columns),
        ],
        sparse_threshold=0.0,
    )


def build_baseline_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(feature_columns=feature_columns)),
            (
                "model",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    solver="saga",
                    tol=1e-3,
                    n_jobs=4,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(feature_columns=feature_columns)),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=4,
                ),
            ),
        ]
    )


def build_extra_trees_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(feature_columns=feature_columns)),
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
    )


def build_xgboost_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "XGBoost candidate requires `pip install xgboost` or requirements-advanced.txt"
        ) from exc
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(feature_columns=feature_columns)),
            (
                "model",
                XGBClassifier(
                    n_estimators=450,
                    max_depth=7,
                    learning_rate=0.045,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=RANDOM_SEED,
                    n_jobs=4,
                ),
            ),
        ]
    )


def build_lightgbm_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LightGBM candidate requires `pip install lightgbm` or requirements-advanced.txt"
        ) from exc
    return Pipeline(
        [
            ("preprocessing", build_preprocessing_pipeline(feature_columns=feature_columns)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.04,
                    num_leaves=48,
                    max_depth=-1,
                    min_child_samples=60,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_alpha=0.1,
                    reg_lambda=2.0,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=4,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_mlp_embeddings_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    from src.models.neural_tabular import TorchTabularClassifier

    return Pipeline(
        [
            (
                "model",
                TorchTabularClassifier(
                    architecture="mlp",
                    hidden_dims=(192, 96, 48),
                    epochs=18,
                    patience=4,
                    batch_size=1024,
                ),
            )
        ]
    )


def build_ft_transformer_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    from src.models.neural_tabular import TorchTabularClassifier

    return Pipeline(
        [
            (
                "model",
                TorchTabularClassifier(
                    architecture="ft_transformer",
                    d_token=24,
                    n_heads=4,
                    n_layers=1,
                    epochs=8,
                    patience=3,
                    batch_size=1024,
                    learning_rate=8e-4,
                ),
            )
        ]
    )


def build_candidate_pipeline(
    candidate_key: str,
    *,
    model_params: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
) -> Pipeline:
    key = MODEL_KEY_ALIASES.get(candidate_key, candidate_key)
    builders = {
        "baseline": build_baseline_pipeline,
        "random_forest": build_random_forest_pipeline,
        "extra_trees": build_extra_trees_pipeline,
        "xgboost": build_xgboost_pipeline,
        "lightgbm": build_lightgbm_pipeline,
        "mlp_embeddings": build_mlp_embeddings_pipeline,
        "ft_transformer": build_ft_transformer_pipeline,
    }
    if key not in builders:
        raise ValueError(f"Unsupported candidate key: {candidate_key}")
    if not candidate_is_available(key):
        dependency = OPTIONAL_DEPENDENCIES[key]
        raise ImportError(
            f"Candidate {key!r} requires optional dependency {dependency!r}. "
            "Install requirements-advanced.txt."
        )
    if key in {"mlp_embeddings", "ft_transformer"} and feature_columns not in (None, FEATURE_COLUMNS):
        raise ValueError(f"{key} currently requires the complete FlightRisk feature schema")
    pipeline = builders[key](feature_columns)
    if model_params:
        pipeline.set_params(**model_params)
    return pipeline


def build_main_pipeline() -> Pipeline:
    return build_extra_trees_pipeline()


def prepare_training_frame(
    train_df: pd.DataFrame,
    *,
    ordered_historical_encoding: bool = True,
    smoothing_strength: float = HistoricalAggregates.DEFAULT_SMOOTHING_STRENGTH,
    schedule_context=None,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, HistoricalAggregates]:
    columns = list(feature_columns or FEATURE_COLUMNS)
    feature_df = add_schedule_features(train_df.copy())
    aggregates = HistoricalAggregates(
        smoothing_strength=smoothing_strength, schedule_context=schedule_context
    )
    if ordered_historical_encoding:
        feature_df = aggregates.fit_transform_ordered(feature_df)
    else:
        aggregates.fit(feature_df)
        feature_df = aggregates.transform(feature_df)
    assert_no_leakage_columns(columns)
    return feature_df[columns].copy(), feature_df[TARGET_COL].astype(int).copy(), aggregates


def fit_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_keys: list[str],
    *,
    parameter_overrides: dict[str, dict[str, Any]] | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, TrainedModel]:
    missing = unavailable_candidates(candidate_keys)
    if missing:
        dependencies = {key: OPTIONAL_DEPENDENCIES[key] for key in missing}
        raise ImportError(
            f"Unavailable FlightRisk candidates: {dependencies}. Install requirements-advanced.txt."
        )
    models: dict[str, TrainedModel] = {}
    for candidate_key in candidate_keys:
        logger.info("Training candidate %s", candidate_key)
        pipeline = build_candidate_pipeline(
            candidate_key,
            model_params=(parameter_overrides or {}).get(candidate_key),
            feature_columns=feature_columns,
        )
        pipeline.fit(X_train, y_train)
        models[candidate_key] = TrainedModel(name=CANDIDATE_NAMES[candidate_key], pipeline=pipeline)
    return models


def train_candidate(
    train_df: pd.DataFrame,
    candidate_key: str,
    *,
    smoothing_strength: float = HistoricalAggregates.DEFAULT_SMOOTHING_STRENGTH,
    model_params: dict[str, Any] | None = None,
    schedule_context=None,
    feature_columns: list[str] | None = None,
) -> tuple[TrainedModel, HistoricalAggregates, pd.DataFrame, pd.Series]:
    key = MODEL_KEY_ALIASES.get(candidate_key, candidate_key)
    columns = list(feature_columns or FEATURE_COLUMNS)
    X_train, y_train, aggregates = prepare_training_frame(
        train_df, ordered_historical_encoding=True, smoothing_strength=smoothing_strength,
        schedule_context=schedule_context, feature_columns=columns,
    )
    models = fit_candidate_models(
        X_train, y_train, [key], parameter_overrides={key: model_params or {}},
        feature_columns=columns,
    )
    return models[key], aggregates, X_train, y_train


def train_models(
    train_df: pd.DataFrame,
    *,
    include_gradient_boosting: bool = False,
    ordered_historical_encoding: bool = True,
    smoothing_strength: float = HistoricalAggregates.DEFAULT_SMOOTHING_STRENGTH,
    candidate_profile: str = "full",
    schedule_context=None,
    feature_columns: list[str] | None = None,
) -> tuple[dict[str, TrainedModel], HistoricalAggregates, pd.DataFrame, pd.Series]:
    keys = candidate_keys_for_profile(
        candidate_profile, include_gradient_boosting=include_gradient_boosting
    )
    columns = list(feature_columns or FEATURE_COLUMNS)
    X_train, y_train, aggregates = prepare_training_frame(
        train_df, ordered_historical_encoding=ordered_historical_encoding,
        smoothing_strength=smoothing_strength, schedule_context=schedule_context,
        feature_columns=columns,
    )
    models = fit_candidate_models(X_train, y_train, keys, feature_columns=columns)
    models["main"] = models.get("random_forest", models["baseline"])
    return models, aggregates, X_train, y_train


def prepare_eval_frame(
    df: pd.DataFrame, aggregates: HistoricalAggregates, *, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    columns = list(feature_columns or FEATURE_COLUMNS)
    feature_df = aggregates.transform(add_schedule_features(df.copy()))
    assert_no_leakage_columns(columns)
    return feature_df[columns].copy(), feature_df[TARGET_COL].astype(int).copy()


def prepare_model_frame(
    df: pd.DataFrame, aggregates: HistoricalAggregates, *, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    return prepare_eval_frame(df, aggregates, feature_columns=feature_columns)
