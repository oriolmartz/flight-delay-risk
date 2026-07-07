"""Thin service layer between the API/dashboard and the trained model artifact."""
from __future__ import annotations

from functools import lru_cache

from src.config import DEFAULT_MODEL_PATH
from src.models.predict import PredictionInput, predict_batch, predict_single, rank_batch
from src.monitoring.prediction_logger import log_prediction
from src.monitoring.monitoring import drift_summary as _drift_summary, prediction_summary as _prediction_summary
from src.models.registry import FlightRiskArtifact
from src.reference.european_layer import (
    build_european_context,
    european_airlines_catalog,
    european_airports_catalog,
)
from src.reference.european_context import lookup_european_context, summarize_european_context, has_real_european_context


@lru_cache(maxsize=1)
def get_artifact() -> FlightRiskArtifact:
    return FlightRiskArtifact.load(DEFAULT_MODEL_PATH)


def is_model_available() -> bool:
    try:
        get_artifact()
        return True
    except FileNotFoundError:
        return False


def model_info() -> dict:
    artifact = get_artifact()
    return {**artifact.metadata, 'metrics': artifact.metrics, 'decision_threshold': artifact.decision_threshold}


def model_card() -> dict:
    artifact = get_artifact()
    metrics = artifact.metrics or {}
    metadata = artifact.metadata or {}
    return {
        'name': 'FlightRisk',
        'version': metadata.get('version', '6.1.0'),
        'task': 'Binary classification: arrival delay of 15+ minutes',
        'target': 'ArrDel15',
        'intended_use': 'Educational portfolio ML system for schedule-time flight delay risk estimation.',
        'not_intended_use': 'Operational aviation, passenger safety, dispatch, or high-stakes travel decisions.',
        'selected_model': metadata.get('model_name', 'unknown'),
        'candidate_models': metadata.get('candidate_models', []),
        'decision_threshold': artifact.decision_threshold,
        'main_metrics': metrics.get('main_model', {}),
        'baseline_metrics': metrics.get('baseline_model', {}),
        'leakage_controls': [
            'Only scheduled/pre-flight fields are used at inference time.',
            'Post-flight delay, taxi, wheels, actual-time and cancellation columns are forbidden as features.',
            'Historical aggregate features are fit on training data only and use fallback values for unseen entities.',
            'Model selection and threshold tuning are performed on validation data, then reported on a held-out test split.',
            'The European layer is a transfer layer over the same model and should be treated as experimental.',
        ],
    }


def predict_flight(payload: PredictionInput, threshold: float | None = None) -> dict:
    artifact = get_artifact()
    effective_threshold = artifact.decision_threshold if threshold is None else threshold
    result = predict_single(artifact, payload, effective_threshold)
    log_prediction(payload, result, artifact.metadata)
    return result


def predict_flights_batch(payloads: list[PredictionInput], threshold: float | None = None) -> list[dict]:
    artifact = get_artifact()
    effective_threshold = artifact.decision_threshold if threshold is None else threshold
    results = predict_batch(artifact, payloads, effective_threshold)
    for payload, result in zip(payloads, results):
        log_prediction(payload, result, artifact.metadata)
    return results


def rank_flights_batch(payloads: list[PredictionInput], threshold: float | None = None) -> dict:
    artifact = get_artifact()
    effective_threshold = artifact.decision_threshold if threshold is None else threshold
    ranked = rank_batch(artifact, payloads, effective_threshold)
    for result in ranked:
        result.setdefault("top_factors", [])
    return {
        "flights_ranked": len(ranked),
        "top_5pct_count": max(1, round(len(ranked) * 0.05)) if ranked else 0,
        "top_10pct_count": max(1, round(len(ranked) * 0.10)) if ranked else 0,
        "ranking_metric_note": "Sorted by predicted ArrDel15 probability; product evaluation should prioritize Precision@TopK and Lift@TopK.",
        "ranked_predictions": ranked,
    }


def predict_european_flight(
    airline: str,
    origin: str,
    destination: str,
    month: int,
    day_of_week: int,
    crs_dep_time: int,
    crs_arr_time: int,
    crs_elapsed_time: int,
    distance: float | None = None,
) -> dict:
    if not has_real_european_context():
        raise ValueError(
            "European mode requires real generated CAA context. Run: "
            "python -m scripts.download_uk_caa_punctuality --year 2024 && "
            "python -m scripts.prepare_uk_caa_context"
        )
    ctx = build_european_context(airline, origin, destination, distance)
    punctuality = lookup_european_context(ctx.airline, ctx.origin, ctx.destination, month)

    payload = PredictionInput(
        airline=ctx.airline,
        origin=ctx.origin,
        destination=ctx.destination,
        month=month,
        day_of_week=day_of_week,
        crs_dep_time=crs_dep_time,
        crs_arr_time=crs_arr_time,
        crs_elapsed_time=crs_elapsed_time,
        distance=ctx.distance_miles,
    )
    result = predict_flight(payload)

    top_factors = list(result.get("top_factors", []))
    if punctuality.pct_flights_15min_late is not None:
        pct = punctuality.pct_flights_15min_late
        if pct >= 0.30:
            top_factors.insert(0, "European route context: elevated historical delay share")
        elif pct <= 0.18:
            top_factors.insert(0, "European route context: comparatively punctual route")
        else:
            top_factors.insert(0, "European route context: near-average punctuality")
        top_factors = top_factors[:4]

    return {
        **result,
        "top_factors": top_factors,
        'region': ctx.region,
        'airline_name': ctx.airline_name,
        'origin_label': ctx.origin_label,
        'destination_label': ctx.destination_label,
        'distance_miles': ctx.distance_miles,
        'distance_source': ctx.distance_source,
        'european_context': punctuality.to_dict(),
        'experimental': True,
        'transfer_note': (
            'European mode combines the BTS-trained flight-level model with an '
            'aggregated European punctuality context layer. Treat it as a portfolio '
            'transfer demo, not a Europe-calibrated operational model.'
        ),
    }


def european_catalog() -> dict:
    return {
        'region': 'europe_experimental',
        'airports': european_airports_catalog(),
        'airlines': european_airlines_catalog(),
        'context_summary': summarize_european_context(),
    }


def european_context_summary() -> dict:
    return summarize_european_context()


def european_route_context(airline: str, origin: str, destination: str, distance: float | None = None) -> dict:
    ctx = build_european_context(airline, origin, destination, distance)
    return {
        'region': ctx.region,
        'airline': ctx.airline,
        'airline_name': ctx.airline_name,
        'origin': ctx.origin,
        'destination': ctx.destination,
        'origin_label': ctx.origin_label,
        'destination_label': ctx.destination_label,
        'distance_miles': ctx.distance_miles,
        'distance_source': ctx.distance_source,
    }



def european_context_for_route(airline: str, origin: str, destination: str, month: int) -> dict:
    return lookup_european_context(airline, origin, destination, month).to_dict()


def prediction_summary() -> dict:
    return _prediction_summary()


def drift_summary() -> dict:
    return _drift_summary()
