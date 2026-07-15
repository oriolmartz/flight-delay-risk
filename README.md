<div align="center">

# Flight Delay Risk

### Pre-departure flight-delay risk workbench · English / Español

Built by **Oriol Martínez**

Flight Delay Risk ranks scheduled flights by estimated arrival-delay exposure, validates every uploaded row, explains the selected model, and exposes the temporal evidence behind the public artifact.

`English / Español` · `FastAPI` · `Streamlit` · `scikit-learn` · `BTS 2024` · `temporal validation` · `calibration` · `PDF reports` · `monitoring` · `Docker`

![Flight Delay Risk product preview](docs/assets/product_preview.svg)

**[English](README.md) · [Español](README_ES.md)**

</div>

> **Core idea.** Most flight-delay demos return a score. Flight Delay Risk turns that score into a review workflow: enter or upload a schedule, rank risk, inspect evidence, verify temporal stability and export a bilingual brief.

## Public release status

**Flight Delay Risk v1.5.0 — Self-Explaining Product UI Release** is the stable portfolio release. It packages a frozen Extra Trees family refitted on a deterministic 250,000-flight sample, an untouched 50,453-flight final test, calibrated top-10% review policy, API/UI, OpenAPI contract, Docker health checks and production smoke evidence.

A hosted URL is not embedded in the archive. The repository is deployment-ready, but the committed smoke proves packaging and runtime contracts rather than external uptime.

The v1.5 interface adopts a very-light-blue aviation palette, a more compact triage banner, qualitative route-support labels, prevalence-aware temporal charts and clearer validation naming. The deployed statistical artifact remains the scaled Extra Trees refit from v1.4; v1.5 changes the public model scope and product surface without reselecting the final model on test evidence.

## Self-explaining interface

The dashboard is designed to be understood without reading this README. Every visible metric includes a one-line interpretation, technical acronyms are translated into product language, and raw diagnostics remain available under **Advanced** expanders.

## Product tour

### 1. Analyze flight

Enter natural schedule fields:

- carrier and optional flight number;
- origin and destination;
- flight date;
- scheduled departure and arrival;
- scheduled duration and distance.

The application derives all model features and returns:

- calibrated probability of a 15+ minute arrival delay;
- raw model score for traceability;
- historical route rate and exact support;
- relative exposure against the route cohort;
- route and carrier-route coverage;
- signed local model contributions;
- bilingual PDF risk brief.

The local explanation is native to the selected estimator. Linear models use exact feature-value × coefficient contributions; tree ensembles use decision-path probability deltas mapped to the ensemble's pre-calibration log-odds change. They explain model behaviour, not real-world causes.

### 2. Rank schedule

Upload the natural CSV template or load the bundled sample:

```csv
flight_number,airline,origin,destination,flight_date,scheduled_departure,scheduled_arrival,scheduled_duration_minutes,distance_miles
418,DL,JFK,LAX,2026-07-18,18:30,21:45,375,2475
```

Flight Delay Risk then:

1. normalizes supported column aliases;
2. validates schema and values row by row;
3. excludes malformed rows without discarding the valid schedule;
4. reports unseen and low-support routes before ranking;
5. transforms the valid batch once;
6. produces calibrated probabilities and historical context;
7. ranks flights into `Priority`, `Watch` and `Routine` queues;
8. exports CSV and bilingual PDF briefs.

Priority tiers are relative to the uploaded schedule. Calibrated probability remains the model's absolute estimate.

### 3. Validation

The validation surface reads committed reports and exposes:

- held-out PR-AUC and Lift@10%;
- Brier score and expected calibration error;
- reliability curve on the untouched test period;
- three expanding temporal folds;
- fold-level model and calibration selection;
- a seven-model benchmark spanning linear, tree, boosting and neural models;
- the ordered historical-encoding contract.

### 4. Model & operations

The operations surface exposes:

- artifact and feature lineage;
- training, validation and test periods;
- live demo prediction counts;
- average logged probability;
- PSI drift status;
- measured local inference latency;
- model card, leakage contract and API surface;
- Docker and public-deployment instructions.

---

## Honest result

### v1.4 scaled refit

The model family and 10% review-capacity policy were frozen before this layer. The release sample grows from 30,000 to **250,000 flights** across all twelve months; 168,519 rows are used for finalist refit, 31,028 for calibration and **50,453 flights** remain in the untouched October 19–December 31 test.

| Metric | v1.5.0 artifact |
|---|---:|
| ROC-AUC | 0.6179 |
| PR-AUC | 0.2386 |
| Precision@Top10% | 0.2801 |
| Lift@Top10% | 1.639× |
| Brier score | 0.1385 |
| Expected calibration error | 0.0130 |

The exact top-10% policy reviews 5,046 flights with precision `0.2800`, recall `0.1639` and lift `1.6384×`. Weekly block intervals on the larger test give PR-AUC `[0.2036, 0.2813]` and Lift@10% `[1.5096, 1.7510]` using 100 resamples. Paired intervals against the SGD numeric logistic baseline exclude zero for PR-AUC, ROC-AUC, Brier and Lift.

The scale-up required an engineering change, not a model-family reselection: Extra Trees now uses compact ordinal `float32` preprocessing, while the baseline is explicitly identified as SGD logistic. A 500,000-row build was attempted but the recency-aware historical encoder exceeded the reproducible resource budget; v1.4 stops at 250,000 rather than claiming unverified scale.

See [`docs/SCALE_REFIT_AND_DEPLOYMENT.md`](docs/SCALE_REFIT_AND_DEPLOYMENT.md).

### Seven-model selection benchmark

The public model zoo is intentionally compact: one interpretable baseline, two bagging ensembles, two modern boosting libraries and two neural tabular models. Every candidate sees the same chronological training and selection blocks with family-appropriate preprocessing.

| Candidate | ROC-AUC | PR-AUC | Lift@10% |
|---|---:|---:|---:|
| **Extra Trees** | **0.6685** | **0.3728** | **1.784×** |
| Random Forest | 0.6633 | 0.3637 | 1.744× |
| Logistic Regression | 0.6486 | 0.3586 | 1.774× |
| LightGBM | 0.6573 | 0.3577 | 1.656× |
| XGBoost | 0.6566 | 0.3524 | 1.665× |
| MLP with embeddings | 0.6481 | 0.3442 | 1.656× |
| FT-Transformer | 0.6416 | 0.3330 | 1.439× |

Extra Trees won the declared PR-AUC selection rule. Random Forest remains as the recognisable bagging reference; XGBoost and LightGBM cover modern boosting; the MLP and FT-Transformer remain first-class neural candidates. Elastic Net, sklearn HistGradientBoosting and CatBoost were removed from the public scope to reduce redundant comparisons.

### Feature-family ablation

The ablation retrains the same Extra Trees configuration on the same chronological blocks. Only the named feature scope changes.

| Scope | Features | PR-AUC | Δ PR-AUC | Lift@10% | Δ Lift |
|---|---:|---:|---:|---:|---:|
| Full system | 112 | **0.3728** | — | 1.784× | — |
| Without core schedule | 80 | 0.3385 | -0.0342 | 1.606× | -0.177× |
| Without calendar | 96 | 0.3667 | -0.0061 | 1.764× | -0.020× |
| Without historical rates | 96 | 0.3686 | -0.0041 | 1.705× | -0.079× |
| Without historical support | 92 | 0.3679 | -0.0048 | 1.803× | +0.020× |
| Without recency | 96 | 0.3704 | -0.0023 | 1.833× | +0.049× |
| Without scheduled congestion | 100 | 0.3689 | -0.0038 | 1.754× | -0.030× |
| Core only | 32 | 0.3602 | -0.0125 | 1.725× | -0.059× |

The full feature system wins the declared PR-AUC objective. Support and recency show a real metric trade-off: removing them lowers PR-AUC but raises point Lift@10%. The report keeps both outcomes instead of labelling every new family a universal improvement.

### Calibration impact

Isotonic calibration was selected on a later holdout inside the calibration block and then refitted on all 3,701 calibration rows.

| Metric | Raw score | Calibrated probability |
|---|---:|---:|
| Brier score | 0.2266 | **0.1409** |
| Expected calibration error | 0.2922 | **0.0304** |
| Log loss | 0.6448 | **0.4576** |
| Mean prediction | 0.4629 | **0.1546** |
| Observed positive rate | 0.1707 | 0.1707 |

Candidate calibrators are fitted on September 5–26, selected on September 27–October 18, and the winning method is refitted on the complete calibration block. The test period remains untouched.

### Temporal validation

The committed three-fold expanding backtest repeats the seven public candidates, fold-local historical feature construction, model selection, calibration-method selection and evaluation.

| Metric | Mean | Std | Range |
|---|---:|---:|---:|
| ROC-AUC | 0.6163 | 0.0537 | 0.5821–0.6783 |
| PR-AUC | 0.2835 | 0.1137 | 0.1942–0.4115 |
| Precision@Top10% | 0.3402 | 0.1192 | 0.2614–0.4774 |
| Lift@Top10% | 1.717× | 0.1321 | 1.577–1.840× |
| Brier score | 0.1523 | 0.0237 | 0.1331–0.1787 |
| ECE | 0.0379 | 0.0310 | 0.0135–0.0727 |

```text
MLP with embeddings selected: 1 / 3 folds
FT-Transformer selected:       1 / 3 folds
Extra Trees selected:          1 / 3 folds
Sigmoid calibration:           2 / 3 folds
Isotonic calibration:          1 / 3 folds
```

The absence of a universal winner is itself a result: representation and model-family performance vary materially across time.

See:

- [`reports/feature_ablation.md`](reports/feature_ablation.md)
- [`reports/candidate_benchmark.md`](reports/candidate_benchmark.md)
- [`reports/temporal_backtest.md`](reports/temporal_backtest.md)
- [`reports/calibration_report.md`](reports/calibration_report.md)

## Current artifact

The canonical parquet contains **6,965,267 cleaned BTS 2024 flights** covering every day of the leap year and fingerprinted in [`data/processed/data_manifest.json`](data/processed/data_manifest.json). The target-free congestion context is fitted on all 6,965,267 schedules.

The v1.5 artifact uses a deterministic **250,000-row proportional sample**:

| Purpose | Dates | Rows |
|---|---|---:|
| Model training | Jan 1 – Jul 16 | 133,599 |
| Prior selection inherited into refit | Jul 17 – Sep 4 | 34,920 |
| Frozen-finalist refit | Jan 1 – Sep 4 | **168,519** |
| Calibration | Sep 5 – Oct 18 | 31,028 |
| Untouched test | Oct 19 – Dec 31 | **50,453** |

Artifact schema v7 records the data fingerprint, schedule context, split dates, sample scale, preprocessing precision, calibrator, operational policy and deployment contract. The packaged model is approximately 52.4 MB.

## Leakage contract

### Allowed before departure

- reporting carrier;
- origin and destination;
- calendar and scheduled times;
- scheduled duration and distance;
- historical rates and frequency maps built from prior dates.

### Explicitly blocked

```text
ArrDelay, ArrDelayMinutes, DepDelay,
ActualElapsedTime, AirTime,
TaxiOut, TaxiIn,
WheelsOff, WheelsOn,
DepTime, ArrTime,
CarrierDelay, WeatherDelay, NASDelay, LateAircraftDelay,
Cancelled, Diverted
```

Training-row historical features use targets from strictly earlier `FlightDate` values. Rows from the same date are transformed together, preventing same-day target leakage.

```python
assert model_train.FlightDate.max() < selection.FlightDate.min()
assert selection.FlightDate.max() < calibration.FlightDate.min()
assert calibration.FlightDate.max() < test.FlightDate.min()
```

---

## Architecture

![Flight Delay Risk architecture](docs/assets/architecture.svg)

```text
BTS data
  -> normalization and leakage removal
  -> source fingerprinting + duplicate-month protection
  -> model-train / selection / calibration / test split
  -> target-free full-timetable schedule context
  -> 112 features: calendar + prior history + support + recency + congestion
  -> candidate comparison on selection only
  -> holdout-selected calibration + threshold
  -> versioned artifact
  -> inference + model-native explanations
  -> bilingual Streamlit / FastAPI / PDF delivery
  -> prediction logging + PSI monitoring
```

Repository map:

```text
app/api/           FastAPI transport and report endpoints
app/dashboard/     bilingual Streamlit product surface
app/services/      prediction, cohort-context and PDF services
src/data/          loading, cleaning and temporal splitting
src/features/      schedule features and historical aggregates
src/models/        training, calibration, explanation and inference
src/monitoring/    prediction logs and PSI drift checks
scripts/           training, backtest, quality and benchmark workflows
reports/           committed model and performance evidence
```

---

## API

```text
GET  /health
GET  /model/info
GET  /model/card
POST /predict
POST /predict/batch
POST /rank
POST /reports/flight
POST /reports/schedule
GET  /monitoring/summary
GET  /monitoring/drift
```

Interactive OpenAPI documentation is available at `/docs` when the API is running.

Example prediction response:

```json
{
  "delay_probability": 0.1691,
  "raw_model_score": 0.5874,
  "calibration_method": "isotonic",
  "risk_level": "moderate",
  "local_contributions": [
    {
      "feature": "RouteDelayRate",
      "contribution": 0.184,
      "direction": "increase"
    }
  ],
  "explanation_scale": "log_odds_before_calibration"
}
```

---

## Measured performance

Committed local release measurements after warm-up:

| Operation | Median |
|---|---:|
| Artifact load | 1,903.4 ms |
| Single prediction | 388.6 ms |
| 100-flight batch | 581.4 ms |
| 1,000-flight batch | 1,543.7 ms |

These figures were measured in the release environment. Hosted latency also depends on network overhead and cold starts. Full environment metadata is in [`reports/performance_benchmark.json`](reports/performance_benchmark.json).

---

## Run locally

### Python

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# For the complete model zoo, including neural and external boosters:
# pip install -r requirements-advanced.txt

streamlit run app/dashboard/streamlit_app.py
uvicorn app.api.main:app --reload
```

### Docker

```bash
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

### Make targets

```bash
make setup-advanced
make test
make quality
make benchmark
make dashboard
make api
```

---

## Reproduce the model evidence

```bash
python -m scripts.prepare_data
python -m scripts.build_schedule_context
python -m scripts.run_feature_ablation --max-rows 30000
python -m scripts.train_model --max-rows 30000 --candidate-profile flagship
python -m scripts.run_temporal_backtest --max-rows 9000 --n-splits 3 --candidate-profile flagship
python -m scripts.run_feature_stability --max-rows 30000 --n-splits 3
python -m scripts.run_policy_backtest --max-rows 30000 --n-splits 3
python -m scripts.build_layer4_release
python -m scripts.evaluate_model
python -m scripts.benchmark_inference
python -m scripts.quality_gate
```

Raw BTS files remain Git-ignored. Download, duplicate-month handling, uniform sampling and manifest generation are documented in [`docs/DATA.md`](docs/DATA.md).

---

## Quality gate

```bash
python -m scripts.quality_gate
```

The v1.5.0 gate verifies:

- compilation and Ruff;
- committed full-suite evidence from the dedicated CI/local test step;
- independent neural training/serialization smoke;
- artifact version and loadability;
- calibrated single and vectorized batch inference;
- model-native explanation output;
- bilingual PDF generation;
- committed temporal, calibration, feature-stability, policy, uncertainty, drift and performance reports;
- release manifest integrity.

---

## Deployment

The repository contains:

- `Dockerfile.api`;
- `Dockerfile.dashboard`;
- `docker-compose.yml`;
- `render.yaml`;
- Streamlit configuration;
- health endpoint and release checks.

Deployment steps and the places where public URLs should be inserted are documented in [`docs/PUBLIC_RELEASE.md`](docs/PUBLIC_RELEASE.md).

---

## Limitations

- No live weather, aircraft rotation, crew, ATC or airport-operations state.
- BTS-trained artifact; the European context layer remains experimental and is not Europe-calibrated.
- Moderate ranking performance reflects a noisy schedule-only problem.
- Historical cohort rates can be weak for unseen or low-support combinations; the UI surfaces both conditions.
- Local contributions explain the selected estimator, not causal mechanisms.
- Monitoring is intentionally lightweight and file-based for a portfolio release.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## What this project demonstrates

- supervised tabular ML on real public records;
- comparison of linear, tree, boosting and neural tabular paradigms;
- learned categorical embeddings and an FT-Transformer with early stopping;
- temporal data splitting and leakage prevention;
- ordered target-rate features with smoothing, support and recency;
- target-free full-timetable congestion context;
- chronological feature-family ablation with negative results preserved;
- honest model selection across time;
- probability calibration;
- ranking metrics for an operational queue;
- model-native local explanation;
- bilingual product design;
- robust CSV onboarding and row-level validation;
- vectorized inference;
- PDF report generation;
- FastAPI, Streamlit, Docker, CI and monitoring;
- the ability to finish and publish an end-to-end ML product.

## License

MIT. Built by **Oriol Martínez**.
