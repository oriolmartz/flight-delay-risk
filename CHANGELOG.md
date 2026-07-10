# Changelog

## v1.0.0 — Public Release

- Completed the bilingual English/Spanish product surface.
- Redesigned the dashboard around a four-step review workflow.
- Added natural CSV onboarding with template download and supported aliases.
- Added row-level validation, partial acceptance of valid rows, unseen-route checks and low-support warnings.
- Added signed local contributions from the deployed L1 Logistic Regression model.
- Clearly separated model explanation from causal interpretation.
- Enriched schedule ranking with route cohort, support, relative exposure and within-schedule percentile.
- Added bilingual single-flight and ranked-schedule PDF reports.
- Added PDF report endpoints to FastAPI.
- Added live demo monitoring, PSI status and committed inference benchmarks to the operations surface.
- Added measured artifact, single-flight and batch latency evidence.
- Added `Makefile`, `render.yaml`, dynamic container ports and a public-release deployment checklist.
- Added a complete Spanish README and rewrote the English README for the final product.
- Expanded the release suite to 84 tests.

## v0.9.0 — Temporal Validation

- Rebuilt historical target-derived features with strictly prior-date ordered encoding.
- Prevented same-day rows from contributing targets to one another.
- Added empirical-Bayes smoothing and exact support counts for historical cohorts.
- Added sigmoid and isotonic post-hoc calibration fitted only on validation data.
- Exposed calibrated probability, raw model score and calibration method separately.
- Added Brier score, log loss and expected calibration error to evaluation reports.
- Added four-fold expanding temporal backtesting with fold-local selection and calibration.
- Retained a separate four-family candidate benchmark for transparent model-development evidence.
- Selected L1 Logistic Regression for the release after it won all four temporal folds.
- Expanded the Validation UI with fold stability, calibration and candidate comparisons.
- Updated the model artifact to schema v2 with date ranges, encoding and calibration lineage.
- Added calibration and temporal-contract tests; the release passed 75 tests.

## v0.8.0 — Product Foundation

- Reframed FlightRisk as a pre-departure schedule-risk workbench.
- Rebuilt the Streamlit interface around Analyze, Rank, Validation and Model & Operations surfaces.
- Added a warm aviation-operations visual system and visible personal byline.
- Replaced numeric calendar/time inputs with natural date and time controls.
- Added historical cohort, coverage and support context to single-flight analysis.
- Converted batch scoring into a prioritised schedule-review queue.
- Prevented identical `FlightDate` values from crossing temporal split boundaries.
- Corrected the L1 Logistic Regression configuration.
- Vectorised batch inference into one model call.
- Introduced a single public product version and a local quality gate.

## Pre-v0.8 development history

Earlier internal versions explored the initial API, model artifact, European transfer layer,
monitoring, ranking metrics and several Streamlit visual directions. v0.8.0 reset the public
versioning scheme around coherent, reviewable portfolio releases.
