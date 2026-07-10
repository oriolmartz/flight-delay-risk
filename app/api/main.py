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
    FlightReportInput,
    HealthResponse,
    ModelCardResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    PredictionOutput,
    RankingOutput,
    RegionCatalogResponse,
    ScheduleReportInput,
)
from app.services import prediction_service, report_service
from src.models.predict import PredictionInput
from src.utils.logging import get_logger
from src.version import APP_VERSION

logger = get_logger(__name__)

app = FastAPI(
    title='FlightRisk API',
    description=(
        'Ranks scheduled flights by arrival-delay risk and returns a post-hoc calibrated estimate of 15+ minute delay, '
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
        version=info.get('version'), release_name=info.get('release_name'),
        artifact_schema_version=info.get('artifact_schema_version'),
        calibration_method=info.get('calibration_method'),
        historical_encoding=info.get('historical_encoding'),
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


@app.post('/reports/flight', tags=['reports'])
def flight_report(flight: FlightReportInput) -> Response:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    payload = _to_prediction_input(flight)
    try:
        prediction = prediction_service.predict_flight(payload)
        context = prediction_service.prediction_context(payload)
        metadata = {
            'airline': flight.airline,
            'flight_number': flight.flight_number or '',
            'origin': flight.origin,
            'destination': flight.destination,
            'flight_date': flight.flight_date or '',
            'scheduled_departure': f'{flight.crs_dep_time:04d}',
            'scheduled_arrival': f'{flight.crs_arr_time:04d}',
            'review_label': prediction.get('risk_level', 'watch'),
        }
        pdf = report_service.build_flight_brief_pdf(metadata, prediction, context, lang=flight.language)
    except Exception as exc:  # pragma: no cover
        logger.exception('Flight report generation failed')
        raise HTTPException(status_code=400, detail=f'Flight report generation failed: {exc}') from exc
    return Response(content=pdf, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename=flightrisk_flight_brief.pdf'})


@app.post('/reports/schedule', tags=['reports'])
def schedule_report(batch: ScheduleReportInput) -> Response:
    if not prediction_service.is_model_available():
        raise HTTPException(status_code=503, detail='No trained model artifact found. Train one first.')
    try:
        payloads = [_to_prediction_input(flight) for flight in batch.flights]
        predictions = prediction_service.predict_flights_batch(payloads)
        contexts = prediction_service.prediction_contexts(payloads)
        import pandas as pd

        rows = []
        for idx, (flight, prediction, context) in enumerate(zip(batch.flights, predictions, contexts), start=1):
            route_rate = float(context.get('route_rate', 0.0))
            probability = float(prediction.get('delay_probability', 0.0))
            rows.append({
                'rank': idx,
                'airline': flight.airline,
                'flight_number': '',
                'origin': flight.origin,
                'destination': flight.destination,
                'crs_dep_time': flight.crs_dep_time,
                'delay_probability': probability,
                'route_rate': route_rate,
                'relative_exposure': probability / route_rate if route_rate > 0 else 0.0,
                'route_support': context.get('route_support', 0),
                'priority_tier': 'Routine',
            })
        frame = pd.DataFrame(rows).sort_values('delay_probability', ascending=False).reset_index(drop=True)
        frame['rank'] = range(1, len(frame) + 1)
        priority_count = max(1, round(len(frame) * 0.10))
        watch_count = max(priority_count, round(len(frame) * 0.30))
        frame.loc[frame['rank'] <= watch_count, 'priority_tier'] = 'Watch'
        frame.loc[frame['rank'] <= priority_count, 'priority_tier'] = 'Priority'
        pdf = report_service.build_schedule_brief_pdf(frame, lang=batch.language)
    except Exception as exc:  # pragma: no cover
        logger.exception('Schedule report generation failed')
        raise HTTPException(status_code=400, detail=f'Schedule report generation failed: {exc}') from exc
    return Response(content=pdf, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename=flightrisk_schedule_brief.pdf'})


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
