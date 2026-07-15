# Flight Delay Risk predictive feature system

Flight Delay Risk v1.3.0 uses **112 pre-departure features** grouped into six auditable families. Every target-derived value is computed from strictly earlier flight dates; scheduled-congestion features are target-free and are fitted from the complete published 2024 timetable.

## Feature families

| Family | Features | Role |
|---|---:|---|
| Core schedule | 32 | Carrier, airports, route, scheduled times, duration, distance, peak/red-eye flags and cyclic time representations |
| Calendar | 16 | Exact calendar position, season, quarter, annual cycles and U.S. federal-holiday proximity |
| Historical rates | 16 | Smoothed long-run delay rates and target-free frequency shares for operational cohorts |
| Historical support | 20 | Exact and log support counts so the model can distinguish sparse from mature cohort estimates |
| Recency | 16 | 28-day, 90-day, EWMA and recent-vs-long trend signals for carrier, route, origin and destination |
| Scheduled congestion | 12 | Target-free departure/arrival density, daily timetable volume and 60-minute bank concentration |

The canonical lists live in `src/config.py` as `FEATURE_FAMILIES`. The model artifact records the complete schema and feature-set identifier.

## Core schedule

The core family contains the natural schedule inputs and deterministic transformations available before departure:

- airline, origin, destination and route;
- month, weekday and weekend flag;
- scheduled departure/arrival hour and minute;
- duration, distance, scheduled speed and log distance;
- departure/arrival period and distance band;
- cyclic hour/minute encodings;
- morning/evening peak, red-eye, long-haul and overnight-schedule flags.

## Calendar intelligence

When `flight_date` is available, Flight Delay Risk derives exact day-of-month, day-of-year, week, quarter, year progress, season and holiday-distance features. API requests may omit the date for backward compatibility; the artifact then uses a documented month/weekday fallback and sets `CalendarDateKnown = 0`.

## Strictly prior historical evidence

Historical maps are built in complete-date order:

1. transform every row dated `t` using targets from dates `< t`;
2. expose no same-day outcomes;
3. update the state only after all rows on date `t` have been transformed.

Long-run rates use empirical-Bayes smoothing toward the global prior. Exact counts and `log1p(count)` are separate model features, not only UI metadata.

## Recency system

For carrier, route, origin and destination, Flight Delay Risk computes:

- smoothed delay rate over the previous 28 days;
- smoothed delay rate over the previous 90 days;
- exponentially weighted delay rate with a 28-day half-life;
- 28-day rate minus the corresponding long-run rate.

All windows exclude the current flight date. Selection, calibration, test and backtest folds receive frozen or fold-local maps from permitted prior partitions only.

## Target-free scheduled congestion

`data/processed/schedule_context.joblib` is fitted from the complete canonical timetable without using `ArrDel15`. It supplies expected schedule density by weekday, airport and time slot:

- origin departures within ±30, ±60 and ±120 minutes;
- destination arrivals within ±30, ±60 and ±120 minutes;
- expected origin/destination daily scheduled volume;
- carrier-origin and route daily volume;
- origin and destination 60-minute bank share.

The same serialized reference is used during training and serving, preventing train/serve skew. Its source-row count and date range are embedded in artifact metadata and the release manifest.

## Candidate-specific representations

The `flagship` profile compares seven candidates through family-appropriate representations:

- Logistic Regression: scaled numerical features plus sparse one-hot categories;
- Random Forest and Extra Trees: sparse engineered matrix for the release benchmark;
- XGBoost and LightGBM: sparse engineered matrix with histogram-based tree growth;
- MLP and FT-Transformer: learned categorical embeddings plus normalized numerical features.

The deployed scaled Extra Trees artifact uses a compact ordinal `float32` representation to control memory use at 250,000 sampled flights. The public benchmark remains a fixed selection experiment and is not reused to reselect the deployed model after test evaluation.

## Feature ablation

The committed ablation retrains Extra Trees on the same chronological model-training and selection blocks, changing only the feature scope.

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

The full system wins the declared PR-AUC criterion on the selection period. Support and recency lower PR-AUC when removed but their removal improves point Lift@10%; that trade-off is retained rather than relabelled as a universal feature win.

## Availability and causality guardrail

The following are never model inputs: actual departure/arrival times, observed delays, taxi/air time, delay causes, cancellation outcomes or diversion outcomes. Historical and congestion features are associative operational context, not causal explanations.
