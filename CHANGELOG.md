# Changelog

## 1.5.0 clarity patch — Self-Explaining Product UI

- Replaced unexplained metric tiles with cards that state what each value means and whether higher or lower is better.
- Reduced the single-flight summary to delay risk, risk versus route and historical support.
- Moved raw model scores, calibrator names, Brier, ECE and full diagnostic tables into advanced expanders.
- Translated PR-AUC, Lift@10%, Brier and ECE into plain-language ranking, priority and probability-reliability concepts.
- Simplified temporal-validation and model-benchmark tables while retaining full technical evidence on demand.
- Added bilingual microcopy throughout flight, schedule, validation and operations surfaces.
- Preserved the trained v1.5.0 artifact and all frozen evaluation evidence.

## 1.5.0 — Public Model Zoo & Blue UI Release

- Renamed the public product to **Flight Delay Risk** while preserving internal artifact compatibility.
- Reduced the public model zoo to Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, MLP embeddings and FT-Transformer.
- Removed Elastic Net, sklearn HistGradientBoosting and CatBoost from active profiles and public dependencies.
- Added a very-light-blue visual system, compact triage banner and qualitative route-support labels.
- Reframed expanding-window results as validation rather than blanket stability.
- Added fold prevalence, PR-AUC/prevalence ratios, model badges and labelled chart points.
- Preserved the scaled Extra Trees artifact and untouched-test evidence from v1.4.

## v1.4.0 — Scaled Refit & Deployment Readiness Release

- Increased the deterministic release sample from 30,000 to 250,000 flights.
- Refit the frozen Extra Trees family on 168,519 pre-calibration rows and evaluated 50,453 untouched final-quarter flights.
- Added compact ordinal `float32` tree preprocessing and an explicitly named SGD numeric logistic scale baseline.
- Added isolated fit checkpoints for native-pool stability and reproducible scale builds.
- Added `/live`, `/ready`, release/request/timing headers and environment-configurable runtime paths.
- Added Docker, Compose and Render readiness health checks, exported OpenAPI and production smoke evidence.
- Raised artifact schema to v7 and documented the unsuccessful 500,000-row encoder attempt.

## v1.3.0 — Temporal Robustness & Operational Policy Release

- froze the v1.2 model-family decision before final-test policy analysis;
- added exact top-k review-capacity and constrained cost-sensitive policies;
- added pre-calibration feature-family stability across expanding folds;
- added paired weekly block-bootstrap confidence intervals;
- added PSI/Jensen–Shannon feature drift and monthly outcome monitoring;
- added a three-fold operational-policy backtest;
- exposed policy metadata through artifact, API and ranking output;
- raised the artifact schema to v6 and expanded the public-release quality gate to 101 tests.

## v1.2.0 — Feature Intelligence & Ablation Release

- Added 112 pre-departure features grouped into six auditable families.
- Added target-free scheduled-congestion context fitted on the complete 2024 timetable.
- Added 28/90-day, EWMA and trend historical features with strictly prior-date encoding.
- Added exact cohort support and log-support features.
- Added chronological drop-one-family ablations that preserve negative results.
- Added schedule-context cache lineage and artifact schema v5.
- Added MIT license and advanced-model CI installation.

## v1.1.0 — Model Zoo & Neural Release

- Added a nine-candidate flagship benchmark spanning Logistic Regression, Elastic Net, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost, an embedding MLP and FT-Transformer.
- Added family-specific preprocessing so linear, native-categorical, sparse boosting and neural models are compared through appropriate representations.
- Implemented sklearn-compatible PyTorch tabular estimators with categorical embeddings, chronological inner validation, early stopping and joblib serialization.
- Added an independent neural smoke workflow covering training, probability validity and serialization round trips.
- Added bounded native thread pools for deterministic CPU execution across tree ensembles, boosters and PyTorch.
- Expanded the temporal backtest to retrain and select all nine candidates independently inside every fold.
- Selected Extra Trees on the main PR-AUC selection block; FT-Transformer ranked second by PR-AUC and first by Lift@10%.
- Preserved the near tie between Extra Trees and calibrated Logistic Regression on the untouched final test.
- Upgraded artifact metadata to record framework versions, selected framework and the complete candidate scope.
- Updated the API, dashboard, reports, diagrams, model card and bilingual documentation to v1.1.0.
- Passed the complete release gate: Ruff, 91 tests, neural smoke, calibrated inference, explanation, bilingual PDF/UI, performance and lineage checks.

## v1.0.0 — Layer 1 technical hardening

- Rebuilt the canonical dataset from all 12 complete BTS 2024 files: 6,965,267 valid flights across all 366 days.
- Removed the duplicate January source and added duplicate-month detection with explicit resolution policies.
- Replaced head truncation with deterministic sampling across complete monthly files.
- Added source, configuration and processed-data SHA-256 fingerprints in `data_manifest.json`.
- Added chunked Parquet preparation, schema/range validation and auditable cleaning totals.
- Replaced the former train/validation/test workflow with model-train, selection, calibration and untouched-test blocks.
- Fixed hyperparameter-search leakage by rebuilding historical target aggregates inside every temporal fold.
- Added chronological holdout selection for identity, sigmoid and isotonic calibration before full calibration-block refit.
- Unified local, real-data and canonical training entry points around one orchestrator.
- Expanded temporal backtesting to all four candidate families by default.
- Added artifact/runtime compatibility checks and data/config lineage metadata.
- Added tree-path local contributions so tree ensembles remain explainable when legitimately selected.
- Removed six stale source-string UI tests and replaced them with data-foundation and temporal-protocol tests.
- Regenerated the model artifact and committed evaluation evidence under the hardened protocol.

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
