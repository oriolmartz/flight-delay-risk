"""Train a European aggregate punctuality model from real UK CAA context."""
from __future__ import annotations

import argparse
import json

from src.models.european_punctuality_model import train_european_aggregate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train European aggregate delay-risk model.")
    parser.add_argument("--context", default="data/europe/europe_punctuality_context.csv")
    parser.add_argument("--model-path", default="models/european_punctuality_model.joblib")
    args = parser.parse_args()
    report = train_european_aggregate_model(args.context, args.model_path)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
