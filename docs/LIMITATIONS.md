# FlightRisk limitations

FlightRisk is an ML engineering portfolio product, not an operational aviation system.

## 1. Missing real-time causes

The model does not include live weather, ATC restrictions, aircraft rotation, crew status or airport queues.

**Impact:** discrimination is moderate and individual-flight errors are unavoidable.

## 2. One data year

The current evidence is limited to BTS 2024.

**Impact:** multi-year structural shifts, unusual seasons and long-term carrier/network changes are not tested.

## 3. Public-size release artifact

The full processed source contains 2.36 million rows, while the release artifact uses a deterministic 300,000-row sample across the complete date range.

**Impact:** rare cohort coverage is lower than in a full-data production run.

## 4. Temporal instability

PR-AUC and calibration vary across the four backtest periods.

**Impact:** one headline metric cannot describe all operating conditions.

## 5. Calibration drift

Isotonic calibration is fitted on one validation period. A later change in delay prevalence can make probabilities less reliable.

**Required production control:** monitor outcomes, Brier score and ECE over time and recalibrate when necessary.

## 6. Historical association is not causation

Carrier, route and airport rates capture historical association and potentially unobserved confounding.

**Impact:** context signals must not be interpreted as causal blame.

## 7. Sparse cohorts

Rare or unseen groups are smoothed toward the global prior.

**Impact:** predictions for low-support routes rely more on broad schedule features than route-specific history.

## 8. Regional transfer

The experimental European layer overlays aggregate context on a U.S.-trained model.

**Impact:** it is not a calibrated individual-flight model for Europe.

## 9. Deployment scope

Docker, CI and monitoring hooks are included, but the release does not include a durable production outcome-join pipeline, authentication, autoscaling or persistent telemetry.
