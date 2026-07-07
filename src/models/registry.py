"""
Model artifact packaging and (de)serialization.

A single ``FlightRiskArtifact`` bundles everything required to make a
prediction on a brand-new flight: the fitted sklearn pipeline (preprocessing
+ model), the exact feature column order, the historical aggregate lookup
tables (with fallback values for unseen carriers/routes/airports), plus
metadata and training metrics for transparency (see ``GET /model/info``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import DEFAULT_DECISION_THRESHOLD, DEFAULT_MODEL_PATH, FEATURE_COLUMNS
from src.features.historical_aggregates import HistoricalAggregates
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FlightRiskArtifact:
    pipeline: Any  # fitted sklearn Pipeline (preprocessing + model)
    historical_aggregates: HistoricalAggregates
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "historical_aggregates": self.historical_aggregates.to_dict(),
            "feature_columns": self.feature_columns,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "decision_threshold": self.decision_threshold,
        }
        joblib.dump(payload, path)
        logger.info("Saved model artifact to %s", path)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "FlightRiskArtifact":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact found at {path}. Train a model first with "
                "'python -m scripts.train_model' or 'python -m scripts.run_local_demo'."
            )
        payload = joblib.load(path)
        return cls(
            pipeline=payload["pipeline"],
            historical_aggregates=HistoricalAggregates.from_dict(payload["historical_aggregates"]),
            feature_columns=payload["feature_columns"],
            metadata=payload.get("metadata", {}),
            metrics=payload.get("metrics", {}),
            decision_threshold=float(payload.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)),
        )


def build_metadata(model_name: str, n_train: int, n_test: int, extra: dict | None = None) -> dict:
    meta = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "feature_columns": list(FEATURE_COLUMNS),
    }
    if extra:
        meta.update(extra)
    return meta
