# FlightRisk architecture

FlightRisk v1.0.0 is a production-shaped ML workbench built around one principle: every risk estimate should carry temporal, calibration and historical-support evidence.

![FlightRisk architecture](assets/architecture.svg)

## End-to-end flow

```text
Official BTS flight records
  → schema normalization and cleaning
  → forbidden post-flight columns removed
  → complete-date train / validation / test split
  → schedule feature engineering
  → strictly prior-date historical encoding for training rows
  → smoothed frozen cohort maps for validation/test/inference
  → candidate training and validation selection
  → validation-only calibrator and threshold fitting
  → held-out test reporting
  → expanding temporal backtest
  → versioned artifact
  → shared prediction service
  → Streamlit, FastAPI and monitoring
```

## Data layer

```text
scripts/download_bts_data.py
scripts/prepare_data.py
src/data/load_data.py
src/data/clean.py
src/data/io.py
src/data/split.py
```

Responsibilities:

- download and normalize BTS monthly data;
- filter cancelled, diverted and target-missing rows;
- remove post-flight leakage columns;
- validate required schema;
- persist processed data;
- create chronological partitions without shared dates.

## Feature layer

```text
src/features/build_features.py
src/features/historical_aggregates.py
```

Responsibilities:

- derive route, time, peak, cyclic and distance features;
- build cohort keys;
- create training-row rates from strictly earlier dates;
- fit smoothed rate maps and exact count maps;
- apply explicit global fallbacks to unseen groups.

### Ordered encoding contract

For each date:

1. transform all rows using accumulated prior history;
2. do not expose any same-day targets;
3. update rate/count maps only after transformation.

This is stricter than fitting one aggregate map over the complete training partition.

## Model layer

```text
src/models/train.py
src/models/calibration.py
src/models/evaluate.py
src/models/thresholding.py
src/models/registry.py
src/models/predict.py
```

Responsibilities:

- build sparse preprocessing pipelines;
- train linear and optional tree candidates;
- select candidates using later validation data;
- fit sigmoid and isotonic calibration candidates;
- tune a threshold on calibrated validation probabilities;
- report discrimination, ranking and calibration metrics;
- package the pipeline, cohort maps, calibrator, threshold and lineage;
- guarantee identical single and vectorized batch inference paths.

## Validation layer

```text
scripts/train_model.py
scripts/run_temporal_backtest.py
reports/metrics.json
reports/calibration_report.json
reports/candidate_benchmark.json
reports/temporal_backtest.json
```

The release contains two complementary forms of evidence:

- a four-family benchmark on one validation block;
- a four-fold expanding backtest that repeats selection and calibration.

The final model is selected using stability evidence rather than only the best single-split score.

## Serving layer

```text
app/services/prediction_service.py
app/api/main.py
app/schemas.py
app/dashboard/streamlit_app.py
```

The service layer owns artifact loading and all prediction logic. The API and dashboard consume the same functions, preventing duplicated inference behavior.

Outputs include:

- calibrated probability;
- raw model score;
- calibration method;
- thresholded risk label;
- cohort context and exact support;
- schedule-context heuristics clearly labeled as non-causal.

## Monitoring and release controls

```text
src/monitoring/
monitoring/prediction_log.csv
scripts/quality_gate.py
.github/workflows/
```

The quality gate compiles the package, runs Ruff and all tests, loads the release artifact, executes calibrated single/batch inference and validates the committed experimental reports.
