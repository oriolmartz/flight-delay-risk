"""Local, model-native explanations for FlightRisk's linear classifier.

The selected public artifact is an L1 logistic-regression pipeline.  This
module exposes signed local log-odds contributions without claiming causality.
For unsupported estimators it returns an empty explanation rather than
silently substituting a heuristic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES
from src.models.registry import FlightRiskArtifact


def _feature_group(transformed_name: str) -> tuple[str, str | None]:
    """Map a transformed sklearn feature name back to its raw feature group."""
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__"), None

    if transformed_name.startswith("cat__"):
        encoded = transformed_name.removeprefix("cat__")
        # Match longest names first so CarrierRoute is not confused with Carrier.
        for raw_name in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{raw_name}_"
            if encoded.startswith(prefix):
                return raw_name, encoded[len(prefix) :]
        return encoded, None

    return transformed_name, None


def local_linear_contributions(
    artifact: FlightRiskArtifact,
    X: pd.DataFrame,
    *,
    top_n: int = 6,
) -> list[list[dict[str, Any]]]:
    """Return grouped signed contributions for each row in ``X``.

    Values are additive contributions to the classifier's log-odds before
    probability calibration.  They explain model behaviour, not real-world
    causes.  Only linear models exposing ``coef_`` are supported.
    """
    pipeline = artifact.pipeline
    if not hasattr(pipeline, "named_steps"):
        return [[] for _ in range(len(X))]

    preprocessing = pipeline.named_steps.get("preprocessing")
    model = pipeline.named_steps.get("model")
    if preprocessing is None or model is None or not hasattr(model, "coef_"):
        return [[] for _ in range(len(X))]

    transformed = preprocessing.transform(X)
    feature_names = list(preprocessing.get_feature_names_out())
    coefficients = np.asarray(model.coef_)[0]
    if len(feature_names) != len(coefficients):
        return [[] for _ in range(len(X))]

    raw_matrix = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    raw_matrix = np.atleast_2d(raw_matrix)

    outputs: list[list[dict[str, Any]]] = []
    for row_index, transformed_row in enumerate(raw_matrix):
        grouped: dict[str, dict[str, Any]] = {}
        for name, value, coefficient in zip(feature_names, transformed_row, coefficients):
            contribution = float(value * coefficient)
            if abs(contribution) < 1e-12:
                continue
            group, category = _feature_group(str(name))
            item = grouped.setdefault(
                group,
                {
                    "feature": group,
                    "contribution": 0.0,
                    "active_category": None,
                    "raw_value": None,
                },
            )
            item["contribution"] += contribution
            if category is not None and abs(value) > 0:
                item["active_category"] = category

        raw_row = X.iloc[row_index]
        for feature, item in grouped.items():
            if feature in raw_row.index:
                raw_value = raw_row[feature]
                if isinstance(raw_value, np.generic):
                    raw_value = raw_value.item()
                item["raw_value"] = raw_value
            contribution = float(item["contribution"])
            item["direction"] = "increase" if contribution >= 0 else "decrease"
            item["magnitude"] = abs(contribution)
            item["contribution"] = round(contribution, 6)

        ordered = sorted(grouped.values(), key=lambda item: item["magnitude"], reverse=True)
        outputs.append(ordered[:top_n])

    return outputs
