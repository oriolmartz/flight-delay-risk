# Changelog

## v7.9.3 — README and portfolio documentation cleanup

- Rewrote README as a portfolio-grade technical overview instead of a version-note collage.
- Added clear architecture and training-pipeline diagrams using Mermaid.
- Documented target definition, leakage controls, feature groups, candidate models and metric rationale.
- Added transparent discussion of Random Forest selection versus Logistic Regression held-out test performance.
- Removed Europe from the main README narrative and focused the project on `P(ArrDel15 = 1)`.

## v7.9.2 — Model card visual fix

- Replaced the prediction profile chart with a custom light horizontal probability split.
- Removed dark code-block rendering from the Architecture and Training Pipeline expanders.
- Added clean technical cards and a readable pipeline flow.
- Kept Europe out of the main UI narrative.
- Preserved the read-time sampling fix for `--max-rows-per-month`.
