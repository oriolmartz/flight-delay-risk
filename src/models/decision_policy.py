"""Operational decision policies for calibrated FlightRisk probabilities.

The classifier estimates risk; an operational policy decides which flights receive
scarce human attention. Policies are fitted only on calibration holdouts and are
kept separate from model selection so capacity and cost assumptions remain auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import precision_score, recall_score


@dataclass(frozen=True)
class PolicyCosts:
    true_positive_value: float = 1.0
    false_positive_cost: float = 0.15
    false_negative_cost: float = 1.0
    true_negative_value: float = 0.0


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    threshold: float
    selected_count: int
    selected_share: float
    precision: float
    recall: float
    lift: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    net_utility: float
    utility_per_flight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _arrays(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an operational policy on an empty sample")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite")
    return y, np.clip(p, 0.0, 1.0)


def threshold_for_capacity(probabilities: Any, fraction: float) -> float:
    """Probability cut-off selecting approximately the highest-risk fraction."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(p) == 0:
        raise ValueError("Cannot derive a capacity threshold from an empty sample")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(np.ceil(len(p) * fraction)))
    ordered = np.sort(p)[::-1]
    return float(ordered[min(k - 1, len(ordered) - 1)])


def select_top_fraction(
    probabilities: Any,
    fraction: float,
    *,
    tie_breaker: Any | None = None,
) -> np.ndarray:
    """Return an exact top-k mask with deterministic secondary-score tie handling."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(np.ceil(len(p) * fraction)))
    if tie_breaker is None:
        secondary = np.arange(len(p), 0, -1, dtype=float)
    else:
        secondary = np.asarray(tie_breaker, dtype=float).reshape(-1)
        if len(secondary) != len(p):
            raise ValueError("tie_breaker must match probabilities")
    order = np.lexsort((-secondary, -p))
    mask = np.zeros(len(p), dtype=bool)
    mask[order[:k]] = True
    return mask


def evaluate_policy_mask(
    y_true: Any,
    probabilities: Any,
    selected: Any,
    *,
    policy_name: str,
    threshold: float,
    costs: PolicyCosts | None = None,
) -> PolicyEvaluation:
    y, p = _arrays(y_true, probabilities)
    mask = np.asarray(selected, dtype=bool).reshape(-1)
    if len(mask) != len(y):
        raise ValueError("selected mask must match y_true")
    costs = costs or PolicyCosts()
    tp = int(np.sum(mask & (y == 1)))
    fp = int(np.sum(mask & (y == 0)))
    fn = int(np.sum(~mask & (y == 1)))
    tn = int(np.sum(~mask & (y == 0)))
    precision = float(precision_score(y, mask.astype(int), zero_division=0))
    recall = float(recall_score(y, mask.astype(int), zero_division=0))
    prevalence = float(y.mean())
    lift = float(precision / prevalence) if prevalence > 0 else 0.0
    utility = (
        tp * costs.true_positive_value
        + tn * costs.true_negative_value
        - fp * costs.false_positive_cost
        - fn * costs.false_negative_cost
    )
    return PolicyEvaluation(
        policy_name=policy_name,
        threshold=float(threshold),
        selected_count=int(mask.sum()),
        selected_share=float(mask.mean()),
        precision=precision,
        recall=recall,
        lift=lift,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        net_utility=float(utility),
        utility_per_flight=float(utility / len(y)),
    )


def evaluate_threshold_policy(
    y_true: Any,
    probabilities: Any,
    threshold: float,
    *,
    policy_name: str = "fixed_threshold",
    costs: PolicyCosts | None = None,
) -> PolicyEvaluation:
    y, p = _arrays(y_true, probabilities)
    return evaluate_policy_mask(
        y,
        p,
        p >= float(threshold),
        policy_name=policy_name,
        threshold=float(threshold),
        costs=costs,
    )


def evaluate_capacity_policy(
    y_true: Any,
    probabilities: Any,
    fraction: float,
    *,
    costs: PolicyCosts | None = None,
    tie_breaker: Any | None = None,
) -> PolicyEvaluation:
    y, p = _arrays(y_true, probabilities)
    threshold = threshold_for_capacity(p, fraction)
    return evaluate_policy_mask(
        y,
        p,
        select_top_fraction(p, fraction, tie_breaker=tie_breaker),
        policy_name=f"top_{int(round(fraction * 100))}pct_capacity",
        threshold=threshold,
        costs=costs,
    )


def tune_cost_sensitive_threshold(
    y_true: Any,
    probabilities: Any,
    *,
    costs: PolicyCosts | None = None,
    thresholds: Iterable[float] | None = None,
    max_selected_fraction: float | None = None,
) -> PolicyEvaluation:
    """Choose the threshold with maximum declared utility on a calibration holdout."""
    y, p = _arrays(y_true, probabilities)
    costs = costs or PolicyCosts()
    if thresholds is None:
        quantiles = np.quantile(p, np.linspace(0.01, 0.99, 99))
        thresholds = np.unique(np.concatenate(([0.0], quantiles, [1.0])))
    best: PolicyEvaluation | None = None
    for threshold in thresholds:
        result = evaluate_threshold_policy(
            y,
            p,
            float(threshold),
            policy_name="cost_sensitive_threshold",
            costs=costs,
        )
        if max_selected_fraction is not None and result.selected_share > max_selected_fraction + 1e-12:
            continue
        if best is None or result.net_utility > best.net_utility:
            best = result
        elif best is not None and result.net_utility == best.net_utility:
            # Prefer the lower-review policy when utility is identical.
            if result.selected_share < best.selected_share:
                best = result
    if best is None:
        raise ValueError("No threshold satisfied the selected-share constraint")
    return best


def build_policy_frontier(
    y_true: Any,
    probabilities: Any,
    *,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30),
    costs: PolicyCosts | None = None,
    tie_breaker: Any | None = None,
) -> list[dict[str, Any]]:
    return [
        evaluate_capacity_policy(
            y_true, probabilities, fraction, costs=costs, tie_breaker=tie_breaker
        ).to_dict()
        for fraction in fractions
    ]
