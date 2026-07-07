"""
Optional MLflow experiment tracking.

The project should remain easy to run on a fresh machine, so MLflow is used
when it is installed and explicitly enabled. If MLflow is missing, the training
pipeline continues normally and writes the standard reports/artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config import MLRUNS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def mlflow_enabled() -> bool:
    """Return True when MLflow tracking should be attempted."""
    return os.getenv("FLIGHTRISK_ENABLE_MLFLOW", "0").lower() in {"1", "true", "yes"}


class MLflowRun:
    """Small context manager that degrades gracefully when MLflow is unavailable."""

    def __init__(self, run_name: str, tags: dict[str, str] | None = None):
        self.run_name = run_name
        self.tags = tags or {}
        self._mlflow = None
        self.active = False

    def __enter__(self) -> "MLflowRun":
        if not mlflow_enabled():
            logger.info("MLflow disabled. Set FLIGHTRISK_ENABLE_MLFLOW=1 to enable tracking.")
            return self
        try:
            import mlflow  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("MLflow requested but unavailable: %s", exc)
            return self

        self._mlflow = mlflow
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri((Path("file://") / MLRUNS_DIR.resolve()).as_posix())
        mlflow.set_experiment("FlightRisk")
        mlflow.start_run(run_name=self.run_name)
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        self.active = True
        logger.info("MLflow tracking enabled: %s", MLRUNS_DIR)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.active and self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if self.active and self._mlflow is not None:
            self._mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        if not (self.active and self._mlflow is not None):
            return
        flat: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                flat[f"{prefix}{key}"] = float(value)
        if flat:
            self._mlflow.log_metrics(flat)

    def log_artifact(self, path: str | Path) -> None:
        if self.active and self._mlflow is not None:
            p = Path(path)
            if p.exists():
                self._mlflow.log_artifact(str(p))
