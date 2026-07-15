"""Model-native local explanations for FlightRisk classifiers.

Linear estimators are explained exactly through feature-value × coefficient
contributions. Tree ensembles are decomposed along each decision path and the
per-split probability changes are averaged across trees, then mapped onto the
ensemble's pre-calibration log-odds change. Both methods describe model
behaviour, not causal effects.
"""
from __future__ import annotations

from math import log
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
        for raw_name in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{raw_name}_"
            if encoded.startswith(prefix):
                return raw_name, encoded[len(prefix) :]
        return encoded, None

    return transformed_name, None


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finalize_grouped_contributions(
    grouped_values: dict[str, float],
    X: pd.DataFrame,
    row_index: int,
    *,
    active_categories: dict[str, str] | None = None,
    top_n: int,
) -> list[dict[str, Any]]:
    raw_row = X.iloc[row_index]
    output: list[dict[str, Any]] = []
    active_categories = active_categories or {}

    for feature, contribution in grouped_values.items():
        if abs(contribution) < 1e-12:
            continue
        raw_value = _python_scalar(raw_row[feature]) if feature in raw_row.index else None
        output.append(
            {
                "feature": feature,
                "contribution": round(float(contribution), 6),
                "active_category": active_categories.get(feature),
                "raw_value": raw_value,
                "direction": "increase" if contribution >= 0 else "decrease",
                "magnitude": abs(float(contribution)),
            }
        )

    output.sort(key=lambda item: item["magnitude"], reverse=True)
    return output[:top_n]


def _linear_contributions(
    preprocessing: Any,
    model: Any,
    X: pd.DataFrame,
    *,
    top_n: int,
) -> list[list[dict[str, Any]]]:
    transformed = preprocessing.transform(X)
    feature_names = list(preprocessing.get_feature_names_out())
    coefficients = np.asarray(model.coef_)[0]
    if len(feature_names) != len(coefficients):
        return [[] for _ in range(len(X))]

    matrix = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    matrix = np.atleast_2d(matrix)
    outputs: list[list[dict[str, Any]]] = []

    for row_index, transformed_row in enumerate(matrix):
        grouped: dict[str, float] = {}
        active_categories: dict[str, str] = {}
        for name, value, coefficient in zip(feature_names, transformed_row, coefficients):
            contribution = float(value * coefficient)
            if abs(contribution) < 1e-12:
                continue
            group, category = _feature_group(str(name))
            grouped[group] = grouped.get(group, 0.0) + contribution
            if category is not None and abs(value) > 0:
                active_categories[group] = category

        outputs.append(
            _finalize_grouped_contributions(
                grouped,
                X,
                row_index,
                active_categories=active_categories,
                top_n=top_n,
            )
        )
    return outputs


def _node_positive_probability(estimator: Any, node_id: int) -> float:
    values = np.asarray(estimator.tree_.value[node_id], dtype=float).reshape(-1)
    total = float(values.sum())
    if total <= 0 or values.size < 2:
        return 0.0
    classes = list(np.asarray(estimator.classes_).tolist())
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return float(values[positive_index] / total)


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return log(clipped / (1.0 - clipped))


def _tree_path_contributions(
    preprocessing: Any,
    model: Any,
    X: pd.DataFrame,
    *,
    top_n: int,
) -> list[list[dict[str, Any]]]:
    """Vectorized tree-path decomposition on the pre-calibration log-odds scale.

    Each reached child node contributes its positive-class probability change
    relative to its parent, assigned to the parent's raw feature group. Sparse
    decision-path matrices make the calculation scale across batch rows.
    """
    from scipy.sparse import csr_matrix

    estimators = list(getattr(model, "estimators_", []))
    if not estimators or isinstance(estimators[0], np.ndarray):
        return [[] for _ in range(len(X))]

    transformed = preprocessing.transform(X)
    feature_names = list(preprocessing.get_feature_names_out())
    transformed_groups = [_feature_group(str(name))[0] for name in feature_names]
    group_names = list(dict.fromkeys(transformed_groups))
    group_to_index = {name: index for index, name in enumerate(group_names)}
    transformed_to_group = np.asarray(
        [group_to_index[name] for name in transformed_groups], dtype=np.int32
    )

    n_rows = len(X)
    grouped_matrix = np.zeros((n_rows, len(group_names)), dtype=float)
    root_probabilities = np.zeros(len(estimators), dtype=float)

    for estimator_index, estimator in enumerate(estimators):
        tree = estimator.tree_
        values = np.asarray(tree.value, dtype=float).reshape(tree.node_count, -1)
        totals = values.sum(axis=1)
        classes = list(np.asarray(estimator.classes_).tolist())
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        probabilities = np.divide(
            values[:, positive_index],
            totals,
            out=np.zeros(tree.node_count, dtype=float),
            where=totals > 0,
        )
        root_probabilities[estimator_index] = probabilities[0]

        parents = np.flatnonzero(tree.feature >= 0)
        if parents.size == 0:
            continue
        children = np.concatenate(
            [tree.children_left[parents], tree.children_right[parents]]
        ).astype(np.int32)
        repeated_parents = np.concatenate([parents, parents]).astype(np.int32)
        split_features = tree.feature[repeated_parents].astype(np.int32)
        group_columns = transformed_to_group[split_features]
        deltas = probabilities[children] - probabilities[repeated_parents]
        node_to_group = csr_matrix(
            (deltas, (children, group_columns)),
            shape=(tree.node_count, len(group_names)),
        )
        grouped_matrix += (estimator.decision_path(transformed) @ node_to_group).toarray()

    grouped_matrix /= float(len(estimators))
    base_probability = float(root_probabilities.mean())
    predicted_probabilities = np.asarray(model.predict_proba(transformed))[:, 1]
    probability_deltas = predicted_probabilities - base_probability
    log_odds_deltas = np.asarray(
        [_logit(probability) - _logit(base_probability) for probability in predicted_probabilities]
    )
    scales = np.divide(
        log_odds_deltas,
        probability_deltas,
        out=np.zeros_like(log_odds_deltas),
        where=np.abs(probability_deltas) > 1e-12,
    )
    grouped_matrix *= scales[:, None]

    outputs: list[list[dict[str, Any]]] = []
    for row_index in range(n_rows):
        grouped = {
            feature: float(grouped_matrix[row_index, group_index])
            for group_index, feature in enumerate(group_names)
            if abs(grouped_matrix[row_index, group_index]) >= 1e-12
        }
        active_categories = {
            feature: str(_python_scalar(X.iloc[row_index][feature]))
            for feature in CATEGORICAL_FEATURES
            if feature in X.columns
        }
        outputs.append(
            _finalize_grouped_contributions(
                grouped,
                X,
                row_index,
                active_categories=active_categories,
                top_n=top_n,
            )
        )
    return outputs



def _neutral_feature_value(pipeline: Any, model: Any, feature: str, X: pd.DataFrame) -> Any:
    if feature in CATEGORICAL_FEATURES:
        return "__UNKNOWN__"

    if hasattr(model, "numeric_means_"):
        return float(model.numeric_means_.get(feature, 0.0))

    preprocessing = pipeline.named_steps.get("preprocessing")
    if preprocessing is not None and hasattr(preprocessing, "named_transformers_"):
        numeric_transformer = preprocessing.named_transformers_.get("num")
        if hasattr(numeric_transformer, "mean_"):
            try:
                index = list(X.select_dtypes(exclude=["object", "category"]).columns).index(feature)
                return float(numeric_transformer.mean_[index])
            except Exception:
                pass
        if hasattr(numeric_transformer, "statistics_"):
            try:
                from src.config import NUMERIC_FEATURES

                return float(numeric_transformer.statistics_[NUMERIC_FEATURES.index(feature)])
            except Exception:
                pass

    values = pd.to_numeric(X[feature], errors="coerce")
    median = values.median()
    return float(median) if pd.notna(median) else 0.0


def _model_agnostic_ablation_contributions(
    pipeline: Any,
    model: Any,
    X: pd.DataFrame,
    *,
    top_n: int,
    max_rows: int = 64,
) -> list[list[dict[str, Any]]]:
    """Local log-odds change after neutralising one raw feature at a time.

    This fallback supports native boosting and neural models. It is deliberately
    capped for large batches because it needs one extra inference call per raw
    feature. Values explain model sensitivity, not causal effects, and do not
    necessarily sum exactly in the presence of interactions.
    """
    if len(X) > max_rows:
        return [[] for _ in range(len(X))]

    original = np.asarray(pipeline.predict_proba(X), dtype=float)[:, 1]
    original_logits = np.asarray([_logit(value) for value in original])
    grouped_per_row: list[dict[str, float]] = [dict() for _ in range(len(X))]

    for feature in X.columns:
        perturbed = X.copy()
        perturbed[feature] = _neutral_feature_value(pipeline, model, feature, X)
        perturbed_probability = np.asarray(pipeline.predict_proba(perturbed), dtype=float)[:, 1]
        deltas = original_logits - np.asarray([_logit(value) for value in perturbed_probability])
        for row_index, delta in enumerate(deltas):
            grouped_per_row[row_index][str(feature)] = float(delta)

    outputs: list[list[dict[str, Any]]] = []
    for row_index, grouped in enumerate(grouped_per_row):
        active_categories = {
            feature: str(_python_scalar(X.iloc[row_index][feature]))
            for feature in CATEGORICAL_FEATURES
            if feature in X.columns
        }
        outputs.append(
            _finalize_grouped_contributions(
                grouped,
                X,
                row_index,
                active_categories=active_categories,
                top_n=top_n,
            )
        )
    return outputs

def local_model_contributions(
    artifact: FlightRiskArtifact,
    X: pd.DataFrame,
    *,
    top_n: int = 6,
) -> list[list[dict[str, Any]]]:
    """Return grouped pre-calibration log-odds contributions for each row."""
    pipeline = artifact.pipeline
    if not hasattr(pipeline, "named_steps"):
        return [[] for _ in range(len(X))]

    preprocessing = pipeline.named_steps.get("preprocessing")
    model = pipeline.named_steps.get("model")
    if model is None:
        return [[] for _ in range(len(X))]
    if preprocessing is not None and hasattr(model, "coef_"):
        return _linear_contributions(preprocessing, model, X, top_n=top_n)
    if (
        preprocessing is not None
        and hasattr(model, "estimators_")
        and hasattr(model, "predict_proba")
    ):
        return _tree_path_contributions(preprocessing, model, X, top_n=top_n)
    return _model_agnostic_ablation_contributions(
        pipeline, model, X, top_n=top_n
    )


# Backward-compatible name used by older callers.
def local_linear_contributions(
    artifact: FlightRiskArtifact,
    X: pd.DataFrame,
    *,
    top_n: int = 6,
) -> list[list[dict[str, Any]]]:
    return local_model_contributions(artifact, X, top_n=top_n)
