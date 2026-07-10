"""
Single-flight and batch inference on top of a loaded FlightRiskArtifact.

This module intentionally builds features the exact same way as training
(``add_schedule_features`` + ``HistoricalAggregates.transform``) so there is
no train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import (
    RISK_LOW_MAX,
    RISK_MODERATE_MAX,
)
from src.features.build_features import add_schedule_features
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

    def to_raw_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Airline": self.airline.strip().upper(),
                    "Origin": self.origin.strip().upper(),
                    "Dest": self.destination.strip().upper(),
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


def top_factors_for_input(row: pd.Series, aggregates_fallback: float) -> list[str]:
    """Produce a small, human-readable list of the most notable risk drivers.

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
        top = ["no strong risk drivers identified; near-average flight profile"]

    return top


def predict_single(artifact: FlightRiskArtifact, payload: PredictionInput, threshold: float) -> dict:
    raw_df = payload.to_raw_frame()
    df = add_schedule_features(raw_df)
    df = artifact.historical_aggregates.transform(df)

    X = df[artifact.feature_columns]
    probability = float(artifact.pipeline.predict_proba(X)[:, 1][0])

    return {
        "delay_probability": round(probability, 4),
        "risk_level": risk_level_from_probability(probability),
        "decision_threshold": threshold,
        "top_factors": top_factors_for_input(df.iloc[0], artifact.historical_aggregates.global_fallback),
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
    probabilities = artifact.pipeline.predict_proba(df[artifact.feature_columns])[:, 1]

    return [
        {
            "delay_probability": round(float(probability), 4),
            "risk_level": risk_level_from_probability(float(probability)),
            "decision_threshold": threshold,
            "top_factors": top_factors_for_input(
                row, artifact.historical_aggregates.global_fallback
            ),
        }
        for probability, (_, row) in zip(probabilities, df.iterrows())
    ]


def rank_predictions(predictions: list[dict]) -> list[dict]:
    """Sort predictions by risk and annotate operational ranking buckets."""
    n = len(predictions)
    ranked = sorted(predictions, key=lambda item: item["delay_probability"], reverse=True)
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
        out.append(
            {
                **item,
                "rank": idx,
                "risk_percentile": round(percentile, 4),
                "risk_bucket": bucket,
            }
        )
    return out


def rank_batch(
    artifact: FlightRiskArtifact, payloads: list[PredictionInput], threshold: float
) -> list[dict]:
    return rank_predictions(predict_batch(artifact, payloads, threshold))
