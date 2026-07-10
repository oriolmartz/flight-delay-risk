"""Train a real European aggregate punctuality model from UK CAA context rows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.reference.european_context import DEFAULT_EUROPE_CONTEXT_PATH, load_european_context

DEFAULT_EUROPE_MODEL_PATH = Path("models/european_punctuality_model.joblib")


@dataclass
class EuropeanModelReport:
    rows: int
    target_positive_rate: float
    roc_auc: float | None
    model_path: str

    def to_dict(self) -> dict:
        return {"rows": self.rows, "target_positive_rate": self.target_positive_rate, "roc_auc": self.roc_auc, "model_path": self.model_path}


def build_training_frame(context_path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH, delay_threshold_min: float = 15.0) -> tuple[pd.DataFrame, pd.Series]:
    df = load_european_context(context_path)
    if df.empty:
        raise ValueError("No real European context data found. Run download + prepare scripts first.")
    df = df.dropna(subset=["avg_arrival_delay_min"])
    if len(df) < 10:
        raise ValueError("Need at least 10 European context rows with avg_arrival_delay_min to train.")
    y = (df["avg_arrival_delay_min"] >= delay_threshold_min).astype(int)
    X = df[["month", "airline", "origin", "destination", "number_flights_matched"]].copy()
    X["number_flights_matched"] = X["number_flights_matched"].fillna(0)
    return X, y


def train_european_aggregate_model(context_path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH, model_path: str | Path = DEFAULT_EUROPE_MODEL_PATH) -> EuropeanModelReport:
    X, y = build_training_frame(context_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    categorical = ["airline", "origin", "destination"]
    numeric = ["month", "number_flights_matched"]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, class_weight="balanced")),
    ])

    roc_auc = None
    if len(y.unique()) == 2 and len(y) >= 30:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probs)) if len(set(y_test)) == 2 else None
    else:
        pipeline.fit(X, y)

    joblib.dump({"pipeline": pipeline, "target": "avg_arrival_delay_min >= 15", "features": list(X.columns)}, model_path)
    return EuropeanModelReport(rows=len(X), target_positive_rate=float(y.mean()), roc_auc=roc_auc, model_path=str(model_path))
