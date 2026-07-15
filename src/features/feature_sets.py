"""Named feature subsets used by reproducible ablations."""
from __future__ import annotations

from src.config import FEATURE_COLUMNS, FEATURE_FAMILIES


def feature_columns_without(*families: str) -> list[str]:
    unknown = set(families) - set(FEATURE_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown feature families: {sorted(unknown)}")
    removed = {column for family in families for column in FEATURE_FAMILIES[family]}
    return [column for column in FEATURE_COLUMNS if column not in removed]


def ablation_feature_sets() -> dict[str, list[str]]:
    sets = {"full": list(FEATURE_COLUMNS)}
    for family in FEATURE_FAMILIES:
        sets[f"without_{family}"] = feature_columns_without(family)
    sets["core_only"] = list(FEATURE_FAMILIES["core_schedule"])
    sets["core_plus_historical"] = (
        list(FEATURE_FAMILIES["core_schedule"])
        + list(FEATURE_FAMILIES["historical_rates"])
    )
    return sets
