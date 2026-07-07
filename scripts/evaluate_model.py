"""
CLI: re-evaluate a saved model artifact against a processed dataset.

Usage:
    python -m scripts.evaluate_model --model models/flightrisk_model.joblib --data data/processed/flights_processed.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_MODEL_PATH, DEFAULT_PROCESSED_PATH
from src.data.io import read_processed_frame
from src.data.split import split_train_test
from src.models.evaluate import evaluate_model
from src.models.registry import FlightRiskArtifact
from src.models.train import prepare_eval_frame
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved FlightRisk model artifact.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    artifact = FlightRiskArtifact.load(args.model)
    df = read_processed_frame(args.data)

    _, test_df = split_train_test(df, test_size=args.test_size)
    X_test, y_test = prepare_eval_frame(test_df.copy(), artifact.historical_aggregates)

    results = evaluate_model(
        artifact.pipeline,
        artifact.metadata.get("model_name", "main"),
        X_test,
        y_test,
        threshold=artifact.decision_threshold,
    )

    print(json.dumps(results["metrics"], indent=2))
    logger.info("Classification report:\n%s", results["classification_report"])


if __name__ == "__main__":
    main()
