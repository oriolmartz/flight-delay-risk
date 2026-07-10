"""
Training pipeline: preprocessing + baseline + main model.

Baseline: Logistic Regression (fast, interpretable, standard baseline).
Main model: RandomForestClassifier (strong tabular performance, exposes
native feature_importances_ for the evaluation report and dashboard
explanations, and ships in scikit-learn with no extra optional
dependency). GradientBoostingClassifier or an optional
XGBoost/LightGBM model can be swapped in via build_main_pipeline().
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
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


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the shared preprocessing step: one-hot for categoricals, scaling for numerics."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    numeric_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_l1_logistic_pipeline() -> Pipeline:
    """Sparse L1 Logistic Regression candidate.

    Useful for high-cardinality route/airport one-hot features: it can select a
    smaller set of route/carrier signals while keeping training fast and
    interpretable.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                LogisticRegression(
                    max_iter=700,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    solver="liblinear",
                    penalty="l1",
                    C=0.35,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """RandomForestClassifier candidate pipeline (native feature_importances_)."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def prepare_model_frame(
    df: pd.DataFrame, aggregates: HistoricalAggregates
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply schedule features + historical aggregates and return (X, y)."""
    df = add_schedule_features(df)
    df = aggregates.transform(df)

    assert_no_leakage_columns(list(df.columns))

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y


def build_extra_trees_pipeline() -> Pipeline:
    """ExtraTreesClassifier candidate pipeline: strong, fast bagged trees for sparse tabular data."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_gradient_boosting_pipeline() -> Pipeline:
    """GradientBoosting candidate using dense one-hot encoded features.

    This provides an optional boosting-based tabular candidate using only scikit-learn,
    without making XGBoost/LightGBM mandatory for the portfolio repo.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_SEED,
                    n_estimators=90,
                    learning_rate=0.05,
                    max_depth=3,
                ),
            ),
        ]
    )


def build_main_pipeline() -> Pipeline:
    """Default main pipeline kept for backwards compatibility with v1/v2 tests."""
    return build_random_forest_pipeline()


def train_models(
    train_df: pd.DataFrame,
    *,
    include_gradient_boosting: bool = False,
) -> tuple[dict[str, TrainedModel], HistoricalAggregates, pd.DataFrame, pd.Series]:
    """Fit historical aggregates + schedule features, then train baseline and main models.

    Returns:
        models: dict of trained candidate models. By default this includes
            logistic_regression, random_forest and extra_trees.
            GradientBoostingClassifier is intentionally opt-in because it
            densifies one-hot encoded categorical features and can take hours
            on real BTS monthly data.
        aggregates: fitted HistoricalAggregates (fit on train_df only)
        X_train, y_train: the training feature matrix / target actually used
    """
    # Historical aggregates must first be built from the raw train_df
    # (before schedule features are added) so Route/Airline/Origin/Dest exist.
    train_df = add_schedule_features(train_df)

    aggregates = HistoricalAggregates().fit(train_df)
    train_df = aggregates.transform(train_df)

    assert_no_leakage_columns(list(train_df.columns))

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COL].astype(int).copy()

    models: dict[str, TrainedModel] = {}

    logger.info("Training baseline model (Logistic Regression)...")
    baseline_pipeline = build_baseline_pipeline()
    baseline_pipeline.fit(X_train, y_train)
    models["baseline"] = TrainedModel(name="logistic_regression", pipeline=baseline_pipeline)

    logger.info("Training candidate model (L1 Logistic Regression)...")
    l1_pipeline = build_l1_logistic_pipeline()
    l1_pipeline.fit(X_train, y_train)
    models["logistic_l1"] = TrainedModel(name="logistic_l1", pipeline=l1_pipeline)

    logger.info("Training candidate model (RandomForestClassifier)...")
    rf_pipeline = build_random_forest_pipeline()
    rf_pipeline.fit(X_train, y_train)
    models["random_forest"] = TrainedModel(name="random_forest", pipeline=rf_pipeline)

    logger.info("Training candidate model (ExtraTreesClassifier)...")
    extra_trees_pipeline = build_extra_trees_pipeline()
    extra_trees_pipeline.fit(X_train, y_train)
    models["extra_trees"] = TrainedModel(name="extra_trees", pipeline=extra_trees_pipeline)

    if include_gradient_boosting:
        logger.warning(
            "Training candidate model (GradientBoostingClassifier). "
            "This can be very slow on real BTS data because it uses dense one-hot features."
        )
        gb_pipeline = build_gradient_boosting_pipeline()
        gb_pipeline.fit(X_train, y_train)
        models["gradient_boosting"] = TrainedModel(name="gradient_boosting", pipeline=gb_pipeline)
    else:
        logger.info(
            "Skipping GradientBoostingClassifier by default. "
            "Use include_gradient_boosting=True or --include-gradient-boosting for slow full experiments."
        )

    # Backwards-compatible alias used by the existing tests and simple demos.
    models["main"] = models["random_forest"]

    return models, aggregates, X_train, y_train


def prepare_eval_frame(
    df: pd.DataFrame, aggregates: HistoricalAggregates
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a held-out (test) dataframe using aggregates fit on the training set only."""
    df = add_schedule_features(df)
    df = aggregates.transform(df)
    assert_no_leakage_columns(list(df.columns))
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y
