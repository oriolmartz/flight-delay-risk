"""Thin service layer between the API/dashboard and the trained model artifact."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AIRPORT_STATION_MAP_PATH,
    AIRPORT_TIMEZONE_MAP_PATH,
    DEFAULT_MODEL_PATH,
    REPORTS_DIR,
    SCHEDULE_CONTEXT_PATH,
    WEATHER_BASE_MODEL_PATH,
    WEATHER_HOURLY_PATH,
    WEATHER_MODEL_PATH,
    WEATHER_UI_SUMMARY_PATH,
)
from src.features.build_features import add_schedule_features
from src.models.explain import local_model_contributions
from src.models.predict import PredictionInput, predict_batch, predict_single, rank_batch
from src.models.registry import FlightRiskArtifact
from src.monitoring.monitoring import drift_summary as _drift_summary
from src.monitoring.monitoring import prediction_summary as _prediction_summary
from src.monitoring.prediction_logger import log_prediction
from src.reference.european_context import (
    has_real_european_context,
    lookup_european_context,
    summarize_european_context,
)
from src.reference.european_layer import (
    build_european_context,
    european_airlines_catalog,
    european_airports_catalog,
)
from src.version import APP_VERSION

from src.weather.model_features import WEATHER_MODEL_FEATURES
from src.weather.ui_analytics import (
    load_weather_ui_summary,
    point_in_time_weather_snapshot,
)


@lru_cache(maxsize=1)
def get_artifact() -> FlightRiskArtifact:
    return FlightRiskArtifact.load(DEFAULT_MODEL_PATH)


@lru_cache(maxsize=1)
def get_weather_base_artifact() -> FlightRiskArtifact:
    """Load the schedule-only companion trained on the weather cohort."""
    return FlightRiskArtifact.load(WEATHER_BASE_MODEL_PATH)


@lru_cache(maxsize=1)
def get_weather_artifact() -> FlightRiskArtifact:
    """Load the optional weather-enhanced artifact without replacing the base model."""
    return FlightRiskArtifact.load(WEATHER_MODEL_PATH)


@lru_cache(maxsize=1)
def _weather_source_frames():
    import pandas as pd

    weather = pd.read_parquet(WEATHER_HOURLY_PATH)
    station_mapping = pd.read_csv(AIRPORT_STATION_MAP_PATH, dtype={"station_id": "string"})
    timezone_mapping = pd.read_csv(AIRPORT_TIMEZONE_MAP_PATH)
    return weather, station_mapping, timezone_mapping


def is_model_available() -> bool:
    try:
        get_artifact()
        return True
    except (FileNotFoundError, RuntimeError, ValueError):
        return False


def readiness_status() -> dict:
    """Return deployment readiness without exposing internal paths."""
    checks = {
        "model_artifact": Path(DEFAULT_MODEL_PATH).exists(),
        "schedule_context": Path(SCHEDULE_CONTEXT_PATH).exists(),
        "metrics_report": (REPORTS_DIR / "metrics.json").exists(),
        "scale_refit_report": (REPORTS_DIR / "scale_refit.json").exists(),
    }
    model: dict = {}
    try:
        artifact = get_artifact()
        model = {
            "name": artifact.metadata.get("model_name", "unknown"),
            "version": artifact.metadata.get("version"),
            "artifact_schema_version": artifact.metadata.get("artifact_schema_version"),
            "trained_rows": artifact.metadata.get("n_train_rows"),
        }
        checks["artifact_load"] = True
        checks["release_version_match"] = artifact.metadata.get("version") == APP_VERSION
        checks["schema_v7_or_newer"] = int(artifact.metadata.get("artifact_schema_version", 0)) >= 7
    except Exception:
        checks["artifact_load"] = False
        checks["release_version_match"] = False
        checks["schema_v7_or_newer"] = False
    return {
        "status": "ready" if all(checks.values()) else "degraded",
        "version": APP_VERSION,
        "checks": checks,
        "model": model,
    }


def model_info() -> dict:
    artifact = get_artifact()
    return {**artifact.metadata, 'metrics': artifact.metrics, 'decision_threshold': artifact.decision_threshold, 'operational_policy': artifact.operational_policy}


def model_card() -> dict:
    artifact = get_artifact()
    metrics = artifact.metrics or {}
    metadata = artifact.metadata or {}
    return {
        'name': 'Flight Delay Risk',
        'version': APP_VERSION,
        'task': 'Binary classification: arrival delay of 15+ minutes',
        'target': 'ArrDel15',
        'intended_use': 'Educational portfolio ML system for schedule-time flight delay risk estimation.',
        'not_intended_use': 'Operational aviation, passenger safety, dispatch, or high-stakes travel decisions.',
        'selected_model': metadata.get('model_name', 'unknown'),
        'candidate_models': metadata.get('candidate_models', []),
        'decision_threshold': artifact.decision_threshold,
        'operational_policy': artifact.operational_policy,
        'calibration_method': metadata.get('calibration_method', 'identity'),
        'historical_encoding': metadata.get('historical_encoding', 'train-fitted aggregates'),
        'main_metrics': metrics.get('main_model', {}),
        'baseline_metrics': metrics.get('baseline_model', {}),
        'leakage_controls': [
            'Only scheduled/pre-flight fields are used at inference time.',
            'Post-flight delay, taxi, wheels, actual-time and cancellation columns are forbidden as features.',
            'Training-row historical and recent-form rates use targets from strictly earlier FlightDate values only.',
            'Scheduled-congestion features are target-free and built from published timetable density.',
            'Validation, test and inference use smoothed maps fitted on permitted prior periods, with explicit unseen fallbacks.',
            'Model selection and threshold tuning are performed on validation data, then reported on a held-out test split.',
            'The European layer is a transfer layer over the same model and should be treated as experimental.',
        ],
    }



def input_catalog() -> dict:
    """Return carrier and airport choices encoded in the current artifact."""
    artifact = get_artifact()
    aggregates = artifact.historical_aggregates
    carriers = sorted(str(value) for value in aggregates.carrier_rates)
    airports = sorted(
        {str(value) for value in aggregates.origin_rates}
        | {str(value) for value in aggregates.dest_rates}
    )
    return {"carriers": carriers, "airports": airports}


def airport_historical_summary() -> list[dict]:
    """Return artifact-backed airport rates and support for spatial exploration.

    ``origin_rate`` is the historical share of flights departing an airport
    that arrived at least 15 minutes late. ``destination_rate`` applies the
    same target to flights arriving at that airport. Rates are the smoothed,
    training-fitted values used by the deployed feature pipeline.
    """
    artifact = get_artifact()
    aggregates = artifact.historical_aggregates
    airports = sorted(
        {str(value) for value in aggregates.origin_rates}
        | {str(value) for value in aggregates.dest_rates}
    )
    fallback = float(aggregates.global_fallback)
    return [
        {
            "airport": airport,
            "origin_rate": float(aggregates.origin_rates.get(airport, fallback)),
            "origin_support": int(aggregates.origin_counts.get(airport, 0)),
            "destination_rate": float(aggregates.dest_rates.get(airport, fallback)),
            "destination_support": int(aggregates.dest_counts.get(airport, 0)),
        }
        for airport in airports
    ]



def is_weather_model_available() -> bool:
    try:
        get_weather_base_artifact()
        get_weather_artifact()
        return True
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, AttributeError):
        return False


def weather_ui_summary() -> dict:
    """Return the compact offline weather landscape used by map and heatmap views."""
    return load_weather_ui_summary(WEATHER_UI_SUMMARY_PATH)


def weather_snapshot(payload: PredictionInput) -> dict:
    """Return leakage-safe origin/destination observations available at the cutoff."""
    try:
        weather, station_mapping, timezone_mapping = _weather_source_frames()
        return point_in_time_weather_snapshot(
            payload.to_raw_frame(),
            weather,
            station_mapping,
            timezone_mapping,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "origin": {"available": False},
            "destination": {"available": False},
            "feature_values": {},
        }


def _score_optional_artifact(
    artifact: FlightRiskArtifact,
    payload: PredictionInput,
    *,
    feature_values: dict[str, object] | None = None,
) -> tuple[float, float, pd.DataFrame]:
    raw = payload.to_raw_frame()
    for column, value in (feature_values or {}).items():
        raw[column] = value
    frame = artifact.historical_aggregates.transform(add_schedule_features(raw))
    X = frame.reindex(columns=artifact.feature_columns)
    medians = artifact.metadata.get("weather_imputation_medians", {})
    for column in artifact.feature_columns:
        if column in WEATHER_MODEL_FEATURES:
            fallback = float(medians.get(column, 0.0))
            X[column] = pd.to_numeric(X[column], errors="coerce").fillna(fallback)
    raw_probability = float(artifact.pipeline.predict_proba(X)[:, 1][0])
    if artifact.probability_calibrator is not None:
        probability = float(artifact.probability_calibrator.transform([raw_probability])[0])
    else:
        probability = raw_probability
    return probability, raw_probability, X


def weather_enhanced_prediction(
    payload: PredictionInput,
    *,
    base_result: dict | None = None,
    snapshot: dict | None = None,
) -> dict:
    """Compare paired ExtraTrees artifacts on the same complete-weather cohort.

    The displayed weather delta is weather-model probability minus its paired
    schedule-only companion. The main deployed score is returned separately and
    remains the primary product prediction.
    """
    snapshot = snapshot or weather_snapshot(payload)
    if not snapshot.get("available"):
        return {
            "available": False,
            "reason": "No point-in-time weather observation is available for both endpoints.",
            "snapshot": snapshot,
        }
    try:
        paired_base = get_weather_base_artifact()
        weather_artifact = get_weather_artifact()
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Paired weather artifacts unavailable: {exc}",
            "snapshot": snapshot,
        }

    paired_base_probability, raw_paired_base, _ = _score_optional_artifact(
        paired_base,
        payload,
    )
    weather_probability, raw_weather, weather_X = _score_optional_artifact(
        weather_artifact,
        payload,
        feature_values=snapshot.get("feature_values", {}),
    )
    base_result = base_result or predict_flight(payload)
    contributions = local_model_contributions(weather_artifact, weather_X, top_n=16)[0]
    weather_contributions = [
        item for item in contributions if str(item.get("feature")) in WEATHER_MODEL_FEATURES
    ][:6]
    return {
        "available": True,
        "deployed_probability": float(base_result["delay_probability"]),
        "paired_base_probability": round(paired_base_probability, 4),
        "weather_probability": round(weather_probability, 4),
        "weather_delta": round(weather_probability - paired_base_probability, 4),
        "raw_paired_base_score": round(raw_paired_base, 4),
        "raw_weather_model_score": round(raw_weather, 4),
        "weather_contributions": weather_contributions,
        "snapshot": snapshot,
        "interpretation": "paired_model_counterfactual_not_causal",
    }

def prediction_context(payload: PredictionInput) -> dict:
    """Expose historical cohort context without claiming causal attribution."""
    artifact = get_artifact()
    aggregates = artifact.historical_aggregates
    frame = add_schedule_features(payload.to_raw_frame())
    transformed = aggregates.transform(frame)
    row = transformed.iloc[0]
    route_key = str(row["Route"])
    carrier_route_key = str(row["CarrierRoute"])
    route_support = int(row.get("RouteHistoryCount", 0) or 0)
    carrier_route_support = int(row.get("CarrierRouteHistoryCount", 0) or 0)

    signals = [
        {
            "label": "Route historical delay rate",
            "value": float(row["RouteDelayRate"]),
            "baseline": float(aggregates.global_fallback),
            "support": route_support,
        },
        {
            "label": "Carrier historical delay rate",
            "value": float(row["CarrierDelayRate"]),
            "baseline": float(aggregates.global_fallback),
            "support": None,
        },
        {
            "label": "Origin-hour historical rate",
            "value": float(row["OriginHourDelayRate"]),
            "baseline": float(aggregates.global_fallback),
            "support": int(row.get("OriginHourHistoryCount", 0) or 0),
        },
        {
            "label": "Destination-hour historical rate",
            "value": float(row["DestHourDelayRate"]),
            "baseline": float(aggregates.global_fallback),
            "support": int(row.get("DestHourHistoryCount", 0) or 0),
        },
    ]
    signals.sort(key=lambda item: abs(item["value"] - item["baseline"]), reverse=True)

    return {
        "route": route_key.replace("_", " → "),
        "global_rate": float(aggregates.global_fallback),
        "route_rate": float(row["RouteDelayRate"]),
        "carrier_rate": float(row["CarrierDelayRate"]),
        "origin_rate": float(row["OriginDelayRate"]),
        "destination_rate": float(row["DestDelayRate"]),
        "route_support": route_support,
        "carrier_route_support": carrier_route_support,
        # Backwards-compatible aliases retained for the existing API/UI contract.
        "route_support_estimate": route_support,
        "carrier_route_support_estimate": carrier_route_support,
        "smoothing_strength": float(aggregates.smoothing_strength),
        "route_seen": route_key in aggregates.route_rates,
        "carrier_route_seen": carrier_route_key in aggregates.carrier_route_rates,
        "signals": signals,
    }



def prediction_contexts(payloads: list[PredictionInput]) -> list[dict]:
    """Vectorized historical cohort context for a batch of flights."""
    if not payloads:
        return []
    artifact = get_artifact()
    aggregates = artifact.historical_aggregates
    import pandas as pd

    raw = pd.concat([payload.to_raw_frame() for payload in payloads], ignore_index=True)
    frame = add_schedule_features(raw)
    transformed = aggregates.transform(frame)
    contexts: list[dict] = []
    for _, row in transformed.iterrows():
        route_key = str(row["Route"])
        carrier_route_key = str(row["CarrierRoute"])
        contexts.append(
            {
                "route": route_key.replace("_", " → "),
                "global_rate": float(aggregates.global_fallback),
                "route_rate": float(row["RouteDelayRate"]),
                "carrier_rate": float(row["CarrierDelayRate"]),
                "origin_rate": float(row["OriginDelayRate"]),
                "destination_rate": float(row["DestDelayRate"]),
                "route_support": int(row.get("RouteHistoryCount", 0) or 0),
                "carrier_route_support": int(row.get("CarrierRouteHistoryCount", 0) or 0),
                "route_seen": route_key in aggregates.route_rates,
                "carrier_route_seen": carrier_route_key in aggregates.carrier_route_rates,
            }
        )
    return contexts

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
    policy = artifact.operational_policy or {}
    capacity_fraction = float(policy.get("capacity_fraction", 0.10))
    return {
        "flights_ranked": len(ranked),
        "top_5pct_count": max(1, round(len(ranked) * 0.05)) if ranked else 0,
        "top_10pct_count": max(1, round(len(ranked) * 0.10)) if ranked else 0,
        "policy_capacity_count": max(1, int(np.ceil(len(ranked) * capacity_fraction))) if ranked else 0,
        "policy_capacity_fraction": capacity_fraction,
        "policy_name": str(policy.get("policy_name", "top_10pct_capacity")),
        "ranking_metric_note": "Sorted by calibrated ArrDel15 probability; the operational policy enforces a declared review-capacity budget.",
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
