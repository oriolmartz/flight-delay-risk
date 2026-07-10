"""FlightRisk FastAPI application."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchFlightInput,
    BatchPredictionOutput,
    DriftResponse,
    EuropeanContextSummaryResponse,
    EuropeanFlightInput,
    EuropeanPredictionOutput,
    FlightInput,
    HealthResponse,
    ModelCardResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    PredictionOutput,
    RankingOutput,
    RegionCatalogResponse,
)
from app.services import prediction_service
from src.models.predict import PredictionInput
from src.utils.logging import get_logger
from src.version import APP_VERSION

logger = get_logger(__name__)

app = FastAPI(
    title='FlightRisk API',
    description=(
        'Ranks scheduled flights by arrival-delay risk and estimates the probability of 15+ minute delay, '
        'using only pre-flight schedule-time information. Includes a European context layer.'
    ),
    version=APP_VERSION,
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


def _to_prediction_input(flight: FlightInput) -> PredictionInput:
    return PredictionInput(
        airline=flight.airline, origin=flight.origin, destination=flight.destination, month=flight.month,
        day_of_week=flight.day_of_week, crs_dep_time=flight.crs_dep_time, crs_arr_time=flight.crs_arr_time,
        crs_elapsed_time=flight.crs_elapsed_time, distance=flight.distance,
    )


@app.get('/health', response_model=HealthResponse, tags=['system'])
def health() -> HealthResponse:
    return HealthResponse(status='ok', model_loaded=prediction_service.is_model_available())


@app.get('/model/info', response_model=ModelInfoResponse, tags=['system'])
def get_model_info() -> ModelInfoResponse:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    info = prediction_service.model_info()
    return ModelInfoResponse(
        model_name=info.get('model_name', 'unknown'), trained_at_utc=info.get('trained_at_utc'),
        n_train_rows=info.get('n_train_rows'), n_test_rows=info.get('n_test_rows'),
        validation_rows=info.get('validation_rows'), feature_columns=info.get('feature_columns', []),
        decision_threshold=info.get('decision_threshold'), metrics=info.get('metrics', {}),
    )


@app.get('/model/card', response_model=ModelCardResponse, tags=['system'])
def get_model_card() -> ModelCardResponse:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    return ModelCardResponse(**prediction_service.model_card())


@app.get('/regions/europe', response_model=RegionCatalogResponse, tags=['regions'])
def get_europe_region_catalog() -> RegionCatalogResponse:
    return RegionCatalogResponse(**prediction_service.european_catalog())


@app.get('/regions/europe/context', response_model=EuropeanContextSummaryResponse, tags=['regions'])
def get_europe_context_summary() -> EuropeanContextSummaryResponse:
    return EuropeanContextSummaryResponse(**prediction_service.european_context_summary())


@app.post('/predict', response_model=PredictionOutput, tags=['prediction'])
def predict(flight: FlightInput) -> PredictionOutput:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    try:
        result = prediction_service.predict_flight(_to_prediction_input(flight))
    except Exception as exc:  # pragma: no cover
        logger.exception('Prediction failed')
        raise HTTPException(status_code=400, detail=f'Prediction failed: {exc}') from exc
    return PredictionOutput(**result)


@app.post('/predict/european', response_model=EuropeanPredictionOutput, tags=['prediction'])
def predict_european(flight: EuropeanFlightInput) -> EuropeanPredictionOutput:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    try:
        result = prediction_service.predict_european_flight(
            airline=flight.airline, origin=flight.origin, destination=flight.destination,
            month=flight.month, day_of_week=flight.day_of_week, crs_dep_time=flight.crs_dep_time,
            crs_arr_time=flight.crs_arr_time, crs_elapsed_time=flight.crs_elapsed_time, distance=flight.distance,
        )
    except Exception as exc:
        logger.exception('European prediction failed')
        raise HTTPException(status_code=400, detail=f'European prediction failed: {exc}') from exc
    return EuropeanPredictionOutput(**result)


@app.post('/predict/batch', response_model=BatchPredictionOutput, tags=['prediction'])
def predict_batch_endpoint(batch: BatchFlightInput) -> BatchPredictionOutput:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    try:
        results = prediction_service.predict_flights_batch([_to_prediction_input(f) for f in batch.flights])
    except Exception as exc:  # pragma: no cover
        logger.exception('Batch prediction failed')
        raise HTTPException(status_code=400, detail=f'Batch prediction failed: {exc}') from exc
    return BatchPredictionOutput(predictions=[PredictionOutput(**r) for r in results])


@app.post('/rank', response_model=RankingOutput, tags=['ranking'])
def rank_flights_endpoint(batch: BatchFlightInput) -> RankingOutput:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    try:
        result = prediction_service.rank_flights_batch([_to_prediction_input(f) for f in batch.flights])
    except Exception as exc:  # pragma: no cover
        logger.exception('Ranking failed')
        raise HTTPException(status_code=400, detail=f'Ranking failed: {exc}') from exc
    return RankingOutput(**result)


@app.get('/monitoring/summary', response_model=MonitoringSummaryResponse, tags=['monitoring'])
def get_monitoring_summary() -> MonitoringSummaryResponse:
    return MonitoringSummaryResponse(**prediction_service.prediction_summary())


@app.get('/monitoring/drift', response_model=DriftResponse, tags=['monitoring'])
def get_drift_summary() -> DriftResponse:
    return DriftResponse(**prediction_service.drift_summary())


@app.get('/favicon.ico', include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get('/', tags=['system'])
def root() -> dict:
    return {
        'message': 'FlightRisk API is running. See /docs for interactive API documentation.',
        'health': '/health', 'model_info': '/model/info', 'model_card': '/model/card',
        'europe_catalog': '/regions/europe', 'predict_european': '/predict/european',
        'monitoring_summary': '/monitoring/summary', 'drift': '/monitoring/drift', 'rank': '/rank',
    }
