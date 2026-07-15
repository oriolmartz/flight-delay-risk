# Flight Delay Risk architecture

Flight Delay Risk v1.3.0 is a production-shaped ML workbench built around one principle: every risk estimate should carry data-lineage, temporal, calibration and historical-support evidence.

![Flight Delay Risk architecture](assets/architecture.svg)

## End-to-end flow

```text
Official BTS monthly records
  → source SHA-256 + duplicate-month resolution
  → chunked schema normalization and cleaning
  → canonical full-year parquet + data manifest
  → model-train / selection / calibration / test split by complete dates
  → full-timetable target-free schedule-context cache
  → fold-local strictly prior-date historical rates, support and recency
  → 112-feature schema grouped into six auditable families
  → candidate-family comparison on selection only
  → winning-family refit on model-train + selection
  → chronological calibration-method holdout
  → calibrator refit + threshold tuning on calibration only
  → one untouched test evaluation
  → expanding temporal backtest
  → versioned artifact with data/config/runtime fingerprints
  → shared prediction and explanation service
  → Streamlit, FastAPI, PDF and monitoring
```

## Data layer

```text
scripts/prepare_data.py
src/data/load_data.py
src/data/clean.py
src/data/manifest.py
src/data/preparation.py
src/data/temporal.py
```

Responsibilities:

- inspect and fingerprint monthly sources;
- reject accidental duplicate months by default;
- sample across complete files rather than taking their first rows;
- process the full year in chunks;
- validate dates, schedule fields and categorical identifiers;
- filter cancelled/diverted records and remove post-flight leakage;
- persist the canonical schema and auditable manifest;
- create strictly ordered four-way temporal partitions.

## Feature layer

```text
src/features/build_features.py
src/features/historical_aggregates.py
src/features/schedule_context.py
src/features/feature_sets.py
scripts/build_schedule_context.py
```

Responsibilities:

- derive exact calendar, route, time, peak, cyclic and distance features;
- build target-free scheduled-density context from the complete timetable;
- create training-row long-run and 28/90-day/EWMA rates from strictly earlier dates;
- fit smoothed rate maps plus exact and log support counts;
- apply explicit global fallbacks to unseen groups;
- expose named feature scopes for chronological ablation.

### Ordered encoding contract

For each training date:

1. transform all rows using accumulated prior history;
2. expose no targets from the same date;
3. update rate/count maps only after every row on that date is transformed.

The same process is rebuilt independently inside every tuning or backtest fold.

## Model layer

```text
src/models/train.py
src/models/neural_tabular.py
src/models/calibration.py
src/models/evaluate.py
src/models/thresholding.py
src/models/explain.py
src/models/registry.py
```

Responsibilities:

- build candidate-specific preprocessing pipelines;
- compare seven candidates across linear, tree-ensemble, boosting and neural-tabular paradigms;
- route each family through appropriate preprocessing rather than one universal representation;
- train embedding MLP and FT-Transformer candidates with chronological inner validation and early stopping;
- select the family on a later chronological block;
- refit the selected family on train + selection;
- choose identity, sigmoid or isotonic calibration on a later holdout;
- refit the winning calibrator on the complete calibration block;
- tune the classification threshold without touching test outcomes;
- report discrimination, ranking and probability quality;
- expose exact linear, tree-path or raw-feature neutralisation contributions;
- package lineage, hashes and runtime versions with the artifact.

## Validation layer

```text
scripts/train_model.py
scripts/tune_hyperparameters.py
scripts/run_temporal_backtest.py
scripts/run_feature_ablation.py
reports/candidate_benchmark.json
reports/calibration_report.json
reports/metrics.json
reports/temporal_backtest.json
reports/feature_ablation.json
```

The canonical runner enforces four separate roles. The ablation runner keeps those train/selection blocks fixed and changes only the named feature family:

1. **model_train** — fit every candidate and fold-local target-derived feature;
2. **selection** — choose the candidate family;
3. **calibration** — choose/refit calibrator and tune threshold;
4. **test** — evaluate the frozen decision once.

The hyperparameter search also rebuilds historical aggregates inside every temporal fold. No target-derived transform is fitted before the split.

## Serving layer

```text
app/services/prediction_service.py
app/api/main.py
app/schemas.py
app/dashboard/streamlit_app.py
```

The API and dashboard share the same artifact-loading, feature-building, prediction and explanation paths, preventing train/serve or interface skew.

## Monitoring and release controls

The quality gate verifies compilation, tests, artifact/runtime compatibility, calibrated inference, local explanations, PDF generation, temporal reports, release hashes and agreement between the artifact's data fingerprint and the canonical data manifest.
