# FlightRisk limitations

FlightRisk is designed as an ML engineering portfolio project, not as an operational aviation product. These limitations are intentionally documented because they are exactly the questions a technical interviewer should ask.

## 1. U.S. training window

The baseline run can be trained on a single BTS month for reproducibility and speed. That is real data, but it does not capture full seasonality.

**Impact**
- Weak coverage of summer/winter effects.
- No explicit holiday season modeling.
- Lower robustness to regime shifts across months.

**Mitigation**
- The pipeline already supports multiple monthly BTS CSVs in `data/raw/`.
- Use `scripts.run_temporal_backtest` to evaluate expanding-window performance.
- A stronger production run should use 6-24 months of BTS data.

## 2. Hyperparameters

The main training script uses fixed candidate models first. That is intentional for fast, reproducible end-to-end validation.

**Impact**
- Model performance may be below what a tuned model can achieve.
- Interviewers can reasonably ask why these hyperparameters were chosen.

**Mitigation**
- v6.5 adds `scripts.tune_hyperparameters` with time-aware randomized search.
- Search results are saved to `reports/hyperparameter_search.json`.

## 3. Evaluation design

The main run uses a time-aware train/validation/test split, not random k-fold. That is appropriate for temporal data, but one split is still just one split.

**Impact**
- Metrics may depend on the chosen time period.
- One month can make performance look better or worse than a longer horizon.

**Mitigation**
- v6.5 adds rolling expanding-window backtesting via `scripts.run_temporal_backtest`.
- v6.5 adds bootstrap confidence intervals for held-out ROC-AUC, PR-AUC and F1.

## 4. European data

The U.S. path is flight-level delay-risk ranking using BTS. The European path uses real UK CAA aggregate punctuality data.

**Impact**
- Europe is not the same target granularity as U.S. BTS.
- It models route/airline punctuality patterns, not individual European flight delay probability.

**Mitigation**
- The UI and README separate the U.S. and European modeling paths.
- A true European flight-level v7 would require an official flight-level source such as EUROCONTROL-style data, with scheduled and actual times aligned.

## 5. Deployment

The repository includes Docker, CI and AWS/ECS-oriented examples, but the clean release does not ship a live public endpoint.

**Impact**
- Deployment is demonstrated as infrastructure readiness, not as a hosted production service.

**Mitigation**
- The next deployment step is a hosted FastAPI + Streamlit service with persistent monitoring storage.
