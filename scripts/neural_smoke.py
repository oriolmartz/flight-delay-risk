"""Isolated CPU smoke test for both FlightRisk neural architectures."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, SAMPLE_CSV_PATH
from src.data.clean import clean_flights
from src.data.load_data import normalize_columns
from src.data.split import split_train_test
from src.models.train import build_candidate_pipeline, prepare_eval_frame, prepare_training_frame
from src.version import APP_VERSION


def run_smoke() -> dict:
    raw = pd.read_csv(SAMPLE_CSV_PATH)
    frame = clean_flights(normalize_columns(raw))
    train, test = split_train_test(frame, test_size=0.25)
    X_train, y_train, aggregates = prepare_training_frame(train.iloc[:1200].copy())
    X_test, _ = prepare_eval_frame(test.iloc[:96].copy(), aggregates)

    configurations = {
        "mlp_embeddings": {
            "model__hidden_dims": (32, 16),
            "model__epochs": 2,
            "model__patience": 2,
            "model__batch_size": 256,
        },
        "ft_transformer": {
            "model__d_token": 16,
            "model__n_heads": 4,
            "model__n_layers": 1,
            "model__epochs": 2,
            "model__patience": 2,
            "model__batch_size": 256,
        },
    }
    results = {}
    for candidate, params in configurations.items():
        started = time.perf_counter()
        pipeline = build_candidate_pipeline(candidate, model_params=params)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)
        if probabilities.shape != (len(X_test), 2):
            raise RuntimeError(f"{candidate} returned invalid probability shape")
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"{candidate} returned non-finite probabilities")
        if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
            raise RuntimeError(f"{candidate} returned probabilities outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"{candidate} probabilities do not sum to one")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{candidate}.joblib"
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            restored_probabilities = restored.predict_proba(X_test)
        if not np.allclose(probabilities, restored_probabilities, atol=1e-6):
            raise RuntimeError(f"{candidate} changed after joblib round-trip")

        estimator = pipeline.named_steps["model"]
        results[candidate] = {
            "status": "passed",
            "rows_train": len(X_train),
            "rows_test": len(X_test),
            "epochs_trained": estimator.n_epochs_trained_,
            "best_validation_loss": estimator.best_validation_loss_,
            "elapsed_seconds": time.perf_counter() - started,
        }

    return {"release": APP_VERSION, "status": "passed", "models": results}


def main() -> None:
    output = run_smoke()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "neural_smoke.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
