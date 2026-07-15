"""
Model artifact packaging and (de)serialization.

A single ``FlightRiskArtifact`` bundles everything required to make a
prediction on a brand-new flight: the fitted sklearn pipeline (preprocessing
+ model), the exact feature column order, the historical aggregate lookup
tables (with fallback values for unseen carriers/routes/airports), plus
metadata and training metrics for transparency (see ``GET /model/info``).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
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
    probability_calibrator: Any | None = None
    operational_policy: dict[str, Any] = field(default_factory=dict)

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
            "probability_calibrator": self.probability_calibrator,
            "operational_policy": self.operational_policy,
        }
        joblib.dump(payload, path)
        logger.info("Saved model artifact to %s", path)
        return path

    @classmethod
    def load(
        cls, path: Path = DEFAULT_MODEL_PATH, *, validate_runtime: bool = True
    ) -> "FlightRiskArtifact":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact found at {path}. Train a model first with "
                "'python -m scripts.train_model' or 'python -m scripts.run_local_demo'."
            )
        payload = joblib.load(path)
        metadata = payload.get("metadata", {})
        if validate_runtime:
            _validate_runtime_versions(metadata)
        return cls(
            pipeline=payload["pipeline"],
            historical_aggregates=HistoricalAggregates.from_dict(payload["historical_aggregates"]),
            feature_columns=payload["feature_columns"],
            metadata=metadata,
            metrics=payload.get("metrics", {}),
            decision_threshold=float(payload.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)),
            probability_calibrator=payload.get("probability_calibrator"),
            operational_policy=payload.get("operational_policy", {}),
        )


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:  # pragma: no cover - defensive metadata path
        return None


def _major_minor(value: str) -> tuple[int, int] | None:
    try:
        major, minor, *_ = value.split(".")
        return int(major), int(minor)
    except Exception:
        return None


def _validate_runtime_versions(metadata: dict[str, Any]) -> None:
    packages = {"scikit-learn": metadata.get("sklearn_version")}
    selected_framework = metadata.get("selected_framework")
    framework_versions = metadata.get("framework_versions", {})
    framework_packages = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "pytorch": "torch",
    }
    if selected_framework in framework_packages:
        package = framework_packages[selected_framework]
        packages[package] = framework_versions.get(package)

    for package, trained_version in packages.items():
        runtime_version = _package_version(package)
        if not trained_version or not runtime_version:
            continue
        trained_mm = _major_minor(str(trained_version))
        runtime_mm = _major_minor(str(runtime_version))
        if trained_mm and runtime_mm and trained_mm != runtime_mm:
            raise RuntimeError(
                "Flight Delay Risk artifact/runtime incompatibility: artifact was trained with "
                f"{package} {trained_version}, current runtime is {runtime_version}. "
                "Install the pinned requirements or retrain the artifact in this environment."
            )


def build_metadata(model_name: str, n_train: int, n_test: int, extra: dict | None = None) -> dict:
    meta = {
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "feature_columns": list(FEATURE_COLUMNS),
        "python_version": platform.python_version(),
        "scikit_learn_version": _package_version("scikit-learn"),
        "sklearn_version": _package_version("scikit-learn"),
        "pandas_version": _package_version("pandas"),
        "joblib_version": _package_version("joblib"),
        "framework_versions": {
            package: _package_version(package)
            for package in ("torch", "xgboost", "lightgbm", "optuna")
            if _package_version(package) is not None
        },
    }
    if extra:
        meta.update(extra)
    return meta
