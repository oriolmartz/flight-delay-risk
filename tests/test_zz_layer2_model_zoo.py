"""Contracts for the intentionally compact public model zoo."""
from __future__ import annotations

from src.models.train import CANDIDATE_NAMES, PROFILE_CHOICES, candidate_keys_for_profile

EXPECTED_FLAGSHIP = [
    "baseline",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp_embeddings",
    "ft_transformer",
]


def test_profiles_expose_linear_bagging_boosting_and_neural_families():
    assert set(PROFILE_CHOICES) >= {
        "full",
        "trees",
        "boosting",
        "neural",
        "flagship",
        "all",
    }
    assert candidate_keys_for_profile("flagship") == EXPECTED_FLAGSHIP
    assert candidate_keys_for_profile("all") == EXPECTED_FLAGSHIP
    assert candidate_keys_for_profile("boosting") == ["baseline", "xgboost", "lightgbm"]
    assert candidate_keys_for_profile("neural") == [
        "baseline",
        "mlp_embeddings",
        "ft_transformer",
    ]


def test_retired_candidates_are_not_in_the_public_registry():
    assert set(CANDIDATE_NAMES) == set(EXPECTED_FLAGSHIP)
    assert "elastic_net" not in CANDIDATE_NAMES
    assert "hist_gradient_boosting" not in CANDIDATE_NAMES
    assert "catboost" not in CANDIDATE_NAMES
