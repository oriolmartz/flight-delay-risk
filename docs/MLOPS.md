# FlightRisk v4 MLOps Notes

FlightRisk v4 adds production-oriented ML engineering around the core model:

- optional MLflow experiment tracking
- validation-based model selection
- validation-based threshold tuning
- final held-out test metrics
- error analysis reports
- prediction logging
- lightweight PSI drift checks
- Dockerized API/dashboard
- AWS ECS/Fargate deployment skeleton

## MLflow

MLflow is optional so the repo remains easy to run.

```bash
FLIGHTRISK_ENABLE_MLFLOW=1 python -m scripts.run_local_demo
mlflow ui --backend-store-uri ./mlruns
```

If MLflow is not installed or the flag is disabled, the pipeline continues and writes the standard artifacts under `reports/` and `models/`.

## Monitoring

Every API/dashboard prediction appends a row to:

```text
monitoring/prediction_log.csv
```

The API exposes:

```text
GET /monitoring/summary
GET /monitoring/drift
```

Drift uses a lightweight Population Stability Index comparison against the training reference saved at:

```text
reports/drift_reference.json
```

This is not a full production observability stack. It is a portfolio-grade post-deployment monitoring layer.
