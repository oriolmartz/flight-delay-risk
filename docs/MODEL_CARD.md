# Model Card — FlightRisk v1.0.0

## Summary

FlightRisk estimates the calibrated probability that a scheduled U.S. domestic flight will arrive at least 15 minutes late (`ArrDel15 = 1`) using only schedule-time and prior-history information.

The deployed artifact is an L1 Logistic Regression pipeline with:

- strictly prior-date historical encoding;
- empirical-Bayes smoothing (`alpha = 50`);
- exact cohort-support maps;
- isotonic calibration fitted on validation data only;
- a validation-selected decision threshold;
- schema and lineage metadata.

## Intended use

- Portfolio demonstration of an end-to-end supervised ML system.
- Relative ranking of scheduled flights for review or analysis.
- Exploration of temporal validation, calibration and sparse-cohort handling.
- Demonstration of shared inference through FastAPI and Streamlit.

## Not intended for

- Aviation safety, dispatch, air-traffic-control or operational decisions.
- Live passenger flight-status information.
- Booking, insurance or connection-planning decisions.
- Evaluation or punishment of carriers, airports or employees.
- Use outside the U.S. BTS domain without new validation and calibration.

## Data

- Source: U.S. DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance.
- Source processed table: 2,360,978 cleaned 2024 flight records.
- Release artifact: deterministic 300,000-row sample spanning the complete date range.

| Partition | Dates | Rows |
|---|---|---:|
| Model training | 2024-01-01 → 2024-08-06 | 189,897 |
| Validation/calibration | 2024-08-07 → 2024-10-06 | 49,622 |
| Held-out test | 2024-10-07 → 2024-12-11 | 60,481 |

Raw monthly files and the full processed parquet are not included in the release distribution.

## Target

`ArrDel15` is 1 when arrival delay is 15 minutes or more and 0 otherwise. Cancelled, diverted or target-missing rows are excluded from the supervised target population.

## Feature availability

Allowed inputs are known before departure:

- carrier, origin, destination;
- calendar values;
- scheduled times, duration and distance;
- schedule-derived features;
- historical cohort features computed from prior dates.

Post-departure actuals and delay outcomes are explicitly forbidden.

## Historical encoding

For model-training rows, cohort rates are generated in date order. A row dated `t` can only use targets from dates `< t`. All rows on a shared date receive features before that date is added to history.

For validation, test and inference, rates come from frozen maps fitted on the model-training period. Sparse rates are smoothed toward the model-training global positive rate and unseen groups receive that fallback.

## Model selection

A four-family single validation benchmark was retained as development evidence. Extra Trees narrowly won that individual split. However, a four-fold expanding temporal backtest selected L1 Logistic Regression in all four folds, so the release favors temporal stability over a single-split win.

## Calibration

The deployed isotonic calibrator is fitted only on validation predictions. It is never fitted on the held-out test period.

Held-out calibration:

| Metric | Raw score | Calibrated probability |
|---|---:|---:|
| Brier score | 0.3036 | 0.1336 |
| Expected calibration error | 0.3947 | 0.0229 |
| Log loss | 0.8178 | 0.4378 |

The API exposes both the raw model score and calibrated probability.

## Held-out performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6023 |
| PR-AUC | 0.2124 |
| F1 | 0.3003 |
| Precision@Top10% | 0.2505 |
| Lift@Top10% | 1.557× |
| Brier score | 0.1336 |
| ECE | 0.0229 |

## Temporal backtest

Across four expanding folds:

- L1 Logistic Regression selected in 4/4 folds.
- Isotonic calibration selected in 4/4 folds.
- Mean PR-AUC: 0.2659.
- Mean Lift@Top10%: 1.6569×.
- Mean Brier score: 0.1498.
- Mean ECE: 0.0550.

The variation across folds is material and should not be hidden.

## Limitations

- The model excludes live weather, ATC, aircraft rotation and crew information.
- The signal is useful for ranking but weak for deterministic classification.
- Calibration may drift when the base delay rate changes.
- Cohort context is associative rather than causal.
- Sparse and unseen combinations depend strongly on the global prior.
- The current release is limited to BTS 2024 and a public-size training sample.

## Monitoring

The service can log predictions and compare current feature distributions against a stored PSI reference. Production use would additionally require observed-outcome joins, calibration drift checks, data-quality alerts and scheduled retraining.
