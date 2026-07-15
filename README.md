<div align="center">

# Flight Delay Risk

### Pre-departure decision support for limited airline operations capacity

**Prioritize the flights that deserve attention first — using calibrated delay risk, historical evidence and temporal validation.**

[English](README.md) · [Español](README_ES.md) · [Dataset](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ) · [API contract](docs/openapi.json)

![Flight Delay Risk dashboard](docs/assets/readme_hero.png)

`Python` · `scikit-learn` · `PyTorch` · `XGBoost` · `LightGBM` · `FastAPI` · `Streamlit` · `Docker`

</div>

## The business problem

Airline operations teams cannot investigate every scheduled departure with the same level of attention. Review capacity is limited, while delay risk changes across routes, carriers, airports, time windows and operating periods.

Flight Delay Risk is designed for a simple operational question:

> **Which scheduled flights should an operations analyst review first before departure?**

The system does not treat a probability as a decision. It separates:

```text
scheduled flight data
→ calibrated delay risk
→ review-capacity policy
→ Priority / Watch / Routine queue
```

The current policy prioritizes the highest-risk **10% of an uploaded schedule**. This is a triage tool, not an automatic claim that a flight will be delayed.

| Product question | Flight Delay Risk response |
|---|---|
| **Who is it designed for?** | Airline operations, network-control or disruption-management analysts. |
| **What decision does it support?** | Which flights deserve limited pre-departure review capacity first. |
| **What is predicted?** | Probability of arriving at least 15 minutes late. |
| **What happens when evidence is weak?** | The UI exposes low-support or unseen routes instead of hiding uncertainty. |
| **What does it not use?** | Live weather, aircraft rotation, crew, ATC or post-departure information. |

## Operational result

The deployed Extra Trees artifact was refitted using **168,519 flights**, calibrated on **31,028 later flights**, and evaluated on an untouched **50,453-flight** October–December 2024 test period.

| Outcome | Result | Plain-language meaning |
|---|---:|---|
| **Priority-list lift** | **1.64×** | The top 10% contains about 64% more delayed flights than random selection. |
| **Priority precision** | **28.0%** | Roughly 28 of every 100 prioritized flights were delayed. |
| **Ranking quality (PR-AUC)** | **0.239** | The model ranks delayed flights above non-delayed flights better than the 16–17% test prevalence baseline. |
| **Calibration error (ECE)** | **0.013** | Predicted probabilities closely matched observed frequencies on the final test. |

These are moderate, honest results for a schedule-only problem. The project preserves negative results, temporal variation and drift rather than presenting the best validation number as guaranteed future performance.

## Product workflow

### 1. Analyze one scheduled flight

Enter carrier, route, date, scheduled times, duration and distance. The application returns:

- calibrated probability of a 15+ minute arrival delay;
- risk compared with the historical route baseline;
- number of prior flights supporting that reference;
- the factors that raised or reduced the model estimate;
- a bilingual PDF brief.


### 2. Rank a schedule under limited capacity

Upload the included CSV template or a valid schedule. The system:

1. validates every row;
2. preserves valid rows when others fail;
3. flags unseen and low-support routes;
4. ranks flights by calibrated risk;
5. assigns `Priority`, `Watch` and `Routine` queues;
6. exports CSV and bilingual PDF reports.

Priority tiers are relative to the uploaded schedule. Calibrated probability remains the model's absolute estimate.

### 3. Inspect temporal evidence

The dashboard explains how model choice, ranking quality and calibration changed across later time windows. No model family dominated every fold.


### 4. Monitor deployment health

The repository includes:

- `/live` and `/ready` endpoints;
- model and release metadata;
- request IDs and processing-time headers;
- OpenAPI export;
- prediction logging and lightweight PSI drift monitoring;
- Docker, Compose and Render configurations;
- a production smoke test covering prediction and ranking contracts.

## Why this is a decision system, not another classifier demo

A high score is not automatically actionable. Flight Delay Risk makes the policy layer explicit:

- **Prediction:** How likely is a 15+ minute arrival delay?
- **Evidence:** How much historical support exists for the route and cohort?
- **Constraint:** Only a limited fraction of flights can be reviewed.
- **Decision:** Which flights enter the priority queue?
- **Guardrail:** Is calibration or feature drift deteriorating?

This separation lets the same model support different operational capacities or cost assumptions without pretending that the model itself knows the business decision.

## Data

**Official source:** U.S. Department of Transportation, Bureau of Transportation Statistics — Reporting Carrier On-Time Performance.

- [Download individual flight records from BTS TranStats](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
- [Dataset overview and field coverage](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD)
- [BTS airline on-time statistics](https://www.transtats.bts.gov/ontime/)

The canonical 2024 dataset contains:

- **7,079,081** source rows from 12 monthly files;
- **6,965,267** cleaned supervised flight records;
- full coverage from January 1 to December 31, including all 366 days;
- a target defined as `ArrDel15 = 1` when arrival is at least 15 minutes late.

Raw BTS files and the processed parquet are intentionally excluded from Git. The committed data manifest records source hashes, cleaning totals, schema, coverage and the processed-dataset fingerprint.

See [`docs/DATA.md`](docs/DATA.md).

## Model comparison

The public model zoo compares recognizable ML paradigms under the same chronological protocol:

| Paradigm | Candidates |
|---|---|
| Interpretable baseline | Logistic Regression |
| Bagging | Random Forest, Extra Trees |
| Gradient boosting | XGBoost, LightGBM |
| Neural tabular | MLP with embeddings, FT-Transformer |

Extra Trees won the declared selection rule. PyTorch is used for the two neural candidates; the deployed model is the scikit-learn Extra Trees ensemble.

<details>
<summary><strong>Selection benchmark</strong></summary>

| Candidate | PR-AUC | Lift@10% |
|---|---:|---:|
| **Extra Trees** | **0.3728** | **1.784×** |
| Random Forest | 0.3637 | 1.744× |
| Logistic Regression | 0.3586 | 1.774× |
| LightGBM | 0.3577 | 1.656× |
| XGBoost | 0.3524 | 1.665× |
| MLP with embeddings | 0.3442 | 1.656× |
| FT-Transformer | 0.3330 | 1.439× |

The model was selected on a chronological selection block, not on the final test.

</details>

## Validation design

The release follows a strict chronological contract:

```text
model training
→ model selection
→ calibration and policy fitting
→ untouched final test
```

Historical target-derived features use only earlier dates. Same-day labels are never used to construct features for another row on that date.

Explicitly blocked post-flight fields include actual delays, actual arrival/departure times, taxi times, cancellation, diversion and delay-cause columns.

<details>
<summary><strong>Technical evidence and reports</strong></summary>

- [`reports/candidate_benchmark.md`](reports/candidate_benchmark.md)
- [`reports/temporal_backtest.md`](reports/temporal_backtest.md)
- [`reports/feature_ablation.md`](reports/feature_ablation.md)
- [`reports/feature_stability.md`](reports/feature_stability.md)
- [`reports/operational_policy.md`](reports/operational_policy.md)
- [`reports/robustness_audit.md`](reports/robustness_audit.md)
- [`reports/drift_analysis.md`](reports/drift_analysis.md)
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)

</details>

## Architecture

![Flight Delay Risk architecture](docs/assets/architecture.svg)

```text
BTS monthly flight records
→ validation, cleaning and source fingerprinting
→ chronological splits
→ schedule, history, support, recency and congestion features
→ model-family comparison
→ calibration
→ top-k operational policy
→ FastAPI / Streamlit / PDF delivery
→ logging, health checks and drift monitoring
```

## Run locally

The trained artifact is included. You do not need to retrain the model to use the application.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Start the dashboard in another terminal:

```bash
python -m streamlit run app/dashboard/streamlit_app.py
```

Open:

- Dashboard: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`
- Readiness: `http://localhost:8000/ready`

Or run both with Docker:

```bash
docker compose up --build
```

## Repository map

```text
app/api/           FastAPI transport and public contracts
app/dashboard/     bilingual Streamlit decision interface
app/services/      prediction and reporting services
src/data/          ingestion, cleaning, manifests and temporal splitting
src/features/      schedule, history, recency and congestion features
src/models/        training, calibration, policy and explanations
src/monitoring/    logs, robustness and drift checks
scripts/           reproducible training, evaluation and release workflows
reports/           committed evidence behind the public artifact
docs/              model card, data guide, deployment and limitations
```

## Limitations

- Schedule-only inputs cannot observe live weather, aircraft rotation, crew, ATC or airport disruption state.
- Ranking performance varies across time; no model family dominated every temporal fold.
- Historical route evidence may be weak for rare or unseen combinations.
- Local feature contributions explain model behaviour, not causal mechanisms.
- The application is deployment-ready, but no public hosted URL is claimed until uptime is verified.

## What this project demonstrates

- end-to-end applied ML engineering on real public records;
- separation of prediction, evidence, policy and action;
- temporal validation and leakage prevention;
- classical, boosting and neural tabular model comparison;
- probability calibration, uncertainty and drift analysis;
- operational ranking under a capacity constraint;
- API, dashboard, PDF reporting, Docker, CI and release evidence;
- honest communication of moderate performance and model limitations.

## License

MIT. Built by **Oriol Martínez**.
