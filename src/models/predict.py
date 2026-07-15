"""
Single-flight and batch inference on top of a loaded FlightRiskArtifact.

This module intentionally builds features the exact same way as training
(``add_schedule_features`` + ``HistoricalAggregates.transform``) so there is
no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    RISK_LOW_MAX,
    RISK_MODERATE_MAX,
)
from src.features.build_features import add_schedule_features
from src.models.explain import local_linear_contributions
from src.models.registry import FlightRiskArtifact


@dataclass
class PredictionInput:
    airline: str
    origin: str
    destination: str
    month: int
    day_of_week: int
    crs_dep_time: int
    crs_arr_time: int
    crs_elapsed_time: int
    distance: float
    flight_date: str | None = None

    def to_raw_frame(self) -> pd.DataFrame:
        flight_date = pd.to_datetime(self.flight_date, errors="coerce") if self.flight_date else pd.NaT
        return pd.DataFrame(
            [
                {
                    "Airline": self.airline.strip().upper(),
                    "Origin": self.origin.strip().upper(),
                    "Dest": self.destination.strip().upper(),
                    "FlightDate": flight_date if pd.notna(flight_date) else pd.NaT,
                    "Year": int(flight_date.year) if pd.notna(flight_date) else 2024,
                    "Month": self.month,
                    "DayOfWeek": self.day_of_week,
                    "CRSDepTime": self.crs_dep_time,
                    "CRSArrTime": self.crs_arr_time,
                    "CRSElapsedTime": self.crs_elapsed_time,
                    "Distance": self.distance,
                }
            ]
        )


def risk_level_from_probability(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "low"
    if probability < RISK_MODERATE_MAX:
        return "moderate"
    return "high"


def operational_decision(artifact: FlightRiskArtifact, probability: float, threshold: float) -> dict:
    policy = artifact.operational_policy or {}
    cutoff = float(policy.get("probability_cutoff", threshold))
    recommended = bool(probability >= cutoff)
    return {
        "review_recommended": recommended,
        "operational_action": "priority_review" if recommended else "standard_monitoring",
        "policy_name": str(policy.get("policy_name", "fixed_threshold")),
        "policy_probability_cutoff": cutoff,
        "policy_capacity_fraction": policy.get("capacity_fraction"),
    }


def top_factors_for_input(row: pd.Series, aggregates_fallback: float) -> list[str]:
    """Produce a small, human-readable list of notable schedule context.

    This is a lightweight, transparent heuristic (not SHAP) intended for a
    recruiter-friendly explanation in the API/dashboard: it surfaces which
    of the pre-flight signals are furthest from "typical".
    """
    factors: list[tuple[str, float]] = []

    if row["DepHour"] >= 18 or row["DepHour"] <= 5:
        factors.append(("evening/night scheduled departure", 1.0))
    if row["IsWeekend"] == 1:
        factors.append(("weekend travel", 0.5))
    if row["RouteDelayRate"] > aggregates_fallback:
        factors.append(("route historical delay rate", row["RouteDelayRate"]))
    if row["CarrierDelayRate"] > aggregates_fallback:
        factors.append(("carrier historical delay rate", row["CarrierDelayRate"]))
    if row["OriginDelayRate"] > aggregates_fallback:
        factors.append(("origin airport historical delay rate", row["OriginDelayRate"]))
    if row["DestDelayRate"] > aggregates_fallback:
        factors.append(("destination airport historical delay rate", row["DestDelayRate"]))
    if row["CRSElapsedTime"] > 240:
        factors.append(("long scheduled flight duration", 0.3))

    factors.sort(key=lambda x: x[1], reverse=True)
    top = [name for name, _ in factors[:3]]

    if not top:
        top = ["no strong schedule-context signals; near-average flight profile"]

    return top


def predict_single(artifact: FlightRiskArtifact, payload: PredictionInput, threshold: float) -> dict:
    raw_df = payload.to_raw_frame()
    df = add_schedule_features(raw_df)
    df = artifact.historical_aggregates.transform(df)

    X = df[artifact.feature_columns]
    raw_probability = float(artifact.pipeline.predict_proba(X)[:, 1][0])
    if artifact.probability_calibrator is not None:
        probability = float(artifact.probability_calibrator.transform([raw_probability])[0])
        calibration_method = artifact.probability_calibrator.method
    else:
        probability = raw_probability
        calibration_method = "identity"

    contributions = local_linear_contributions(artifact, X, top_n=6)[0]

    return {
        "delay_probability": round(probability, 4),
        "raw_model_score": round(raw_probability, 4),
        "calibration_method": calibration_method,
        "risk_level": risk_level_from_probability(probability),
        "decision_threshold": threshold,
        "top_factors": top_factors_for_input(df.iloc[0], artifact.historical_aggregates.global_fallback),
        "local_contributions": contributions,
        "explanation_scale": "log_odds_before_calibration",
        **operational_decision(artifact, probability, threshold),
    }


def predict_batch(
    artifact: FlightRiskArtifact, payloads: list[PredictionInput], threshold: float
) -> list[dict]:
    """Score a batch with one feature transformation and one model call."""
    if not payloads:
        return []

    raw_df = pd.concat([payload.to_raw_frame() for payload in payloads], ignore_index=True)
    df = add_schedule_features(raw_df)
    df = artifact.historical_aggregates.transform(df)
    raw_probabilities = artifact.pipeline.predict_proba(df[artifact.feature_columns])[:, 1]
    if artifact.probability_calibrator is not None:
        probabilities = artifact.probability_calibrator.transform(raw_probabilities)
        calibration_method = artifact.probability_calibrator.method
    else:
        probabilities = raw_probabilities
        calibration_method = "identity"

    explanations = local_linear_contributions(artifact, df[artifact.feature_columns], top_n=6)

    return [
        {
            "delay_probability": round(float(probability), 4),
            "raw_model_score": round(float(raw_probability), 4),
            "calibration_method": calibration_method,
            "risk_level": risk_level_from_probability(float(probability)),
            "decision_threshold": threshold,
            "top_factors": top_factors_for_input(
                row, artifact.historical_aggregates.global_fallback
            ),
            "local_contributions": contributions,
            "explanation_scale": "log_odds_before_calibration",
            **operational_decision(artifact, float(probability), threshold),
        }
        for raw_probability, probability, (_, row), contributions in zip(
            raw_probabilities, probabilities, df.iterrows(), explanations
        )
    ]


def rank_predictions(predictions: list[dict], capacity_fraction: float = 0.10) -> list[dict]:
    """Sort predictions and enforce the declared review-capacity budget."""
    n = len(predictions)
    ranked = sorted(
        predictions,
        key=lambda item: (item["delay_probability"], item.get("raw_model_score", 0.0)),
        reverse=True,
    )
    capacity_count = max(1, int(np.ceil(n * capacity_fraction))) if n else 0
    out: list[dict] = []
    for idx, item in enumerate(ranked, start=1):
        percentile = idx / max(n, 1)
        if percentile <= 0.05:
            bucket = "top_5pct"
        elif percentile <= 0.10:
            bucket = "top_10pct"
        elif percentile <= 0.20:
            bucket = "top_20pct"
        else:
            bucket = "standard_watch"
        policy_selected = idx <= capacity_count
        out.append(
            {
                **item,
                "rank": idx,
                "risk_percentile": round(percentile, 4),
                "risk_bucket": bucket,
                "review_recommended": policy_selected,
                "operational_action": "priority_review" if policy_selected else "standard_monitoring",
            }
        )
    return out


def rank_batch(
    artifact: FlightRiskArtifact, payloads: list[PredictionInput], threshold: float
) -> list[dict]:
    capacity_fraction = float((artifact.operational_policy or {}).get("capacity_fraction", 0.10))
    return rank_predictions(
        predict_batch(artifact, payloads, threshold),
        capacity_fraction=capacity_fraction,
    )
