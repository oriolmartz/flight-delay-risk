"""Integration test: the full train -> evaluate pipeline runs on a small sample dataset."""
from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS, FORBIDDEN_LEAKAGE_COLUMNS, SAMPLE_CSV_PATH, TARGET_COL
from src.data.clean import clean_flights
from src.data.load_data import normalize_columns
from src.data.split import split_train_test
from src.models.evaluate import evaluate_model
from src.models.train import prepare_eval_frame, train_models


def _load_sample() -> pd.DataFrame:
    raw_df = pd.read_csv(SAMPLE_CSV_PATH)
    raw_df = normalize_columns(raw_df)
    return clean_flights(raw_df)


class TestTrainingPipeline:
    def test_sample_data_loads_and_cleans(self):
        df = _load_sample()
        assert len(df) > 0
        assert TARGET_COL in df.columns
        assert df[TARGET_COL].isna().sum() == 0

    def test_no_leakage_columns_survive_cleaning(self):
        df = _load_sample()
        leaked = set(df.columns) & set(FORBIDDEN_LEAKAGE_COLUMNS)
        assert not leaked, f"Leakage columns present after cleaning: {leaked}"

    def test_train_models_runs_end_to_end(self):
        df = _load_sample()
        train_df, test_df = split_train_test(df, test_size=0.25)

        models, aggregates, X_train, y_train = train_models(train_df.copy())

        assert "baseline" in models
        assert "main" in models
        assert list(X_train.columns) == FEATURE_COLUMNS
        assert len(y_train) == len(X_train)

        # No forbidden columns should have made it into the training feature matrix.
        leaked = set(X_train.columns) & set(FORBIDDEN_LEAKAGE_COLUMNS)
        assert not leaked

        X_test, y_test = prepare_eval_frame(test_df.copy(), aggregates)
        assert len(X_test) > 0

        results = evaluate_model(models["main"].pipeline, models["main"].name, X_test, y_test)
        assert 0.0 <= results["metrics"]["roc_auc"] <= 1.0
        assert "classification_report" in results
        assert len(results["confusion_matrix"]) == 2

    def test_predict_proba_returns_valid_probabilities(self):
        df = _load_sample()
        train_df, test_df = split_train_test(df, test_size=0.25)
        models, aggregates, _, _ = train_models(train_df.copy())
        X_test, _ = prepare_eval_frame(test_df.copy(), aggregates)

        proba = models["main"].pipeline.predict_proba(X_test)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()


def test_gradient_boosting_is_opt_in():
    df = _load_sample()
    train_df, _ = split_train_test(df, test_size=0.25)

    models, _, _, _ = train_models(train_df.copy())
    assert "gradient_boosting" not in models

    models_with_gb, _, _, _ = train_models(train_df.copy(), include_gradient_boosting=True)
    assert "gradient_boosting" in models_with_gb
