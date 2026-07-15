# Deployment readiness

## Required runtime files

- `models/flightrisk_model.joblib`
- `data/processed/schedule_context.joblib`
- `reports/metrics.json`
- `reports/scale_refit.json`

`GET /ready` returns HTTP 200 only when these dependencies exist, the artifact loads, the release version matches the application and artifact schema is at least v7.

## Environment variables

```bash
FLIGHTRISK_MODEL_PATH=models/flightrisk_model.joblib
FLIGHTRISK_SCHEDULE_CONTEXT_PATH=data/processed/schedule_context.joblib
FLIGHTRISK_PREDICTION_LOG_PATH=monitoring/predictions.jsonl
PORT=8000
```

## Local container check

```bash
docker compose up --build
curl -f http://localhost:8000/live
curl -f http://localhost:8000/ready
curl -f http://localhost:8000/openapi.json
```

The dashboard waits for the API readiness check in Docker Compose. Render uses `/ready` for the API and `/_stcore/health` for Streamlit.

## Release verification

```bash
python -m scripts.export_openapi
python -m scripts.production_smoke
python -m scripts.quality_gate
```

The smoke report is stored at `reports/production_smoke.json`; the exported contract is `docs/openapi.json`.
