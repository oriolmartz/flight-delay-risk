# Changelog

## v0.8.0 — Product Foundation

- Reframed FlightRisk as a pre-departure schedule-risk workbench.
- Rebuilt the Streamlit interface around Analyze, Rank, Validation and Model & Operations surfaces.
- Added a warm aviation-operations visual system and visible personal byline.
- Replaced numeric calendar/time inputs with natural date and time controls.
- Added historical cohort, coverage and estimated-support context to single-flight analysis.
- Converted batch scoring into a prioritised schedule-review queue.
- Surfaced real held-out metrics and calibration limitations in the UI.
- Prevented identical `FlightDate` values from crossing temporal split boundaries.
- Corrected the L1 Logistic Regression candidate to use `penalty="l1"` explicitly.
- Vectorised batch inference into one model call.
- Introduced a single public product version and cleaned artifact metadata.
- Replaced brittle historical UI assertions with product-contract tests.
- Rewrote the README for recruiters and technical reviewers.
- Added a local quality gate.

## Pre-v0.8 development history

Earlier internal versions explored the initial API, model artifact, European transfer layer,
monitoring, ranking metrics and several Streamlit visual directions. v0.8.0 resets the public
versioning scheme around coherent, reviewable portfolio releases.
