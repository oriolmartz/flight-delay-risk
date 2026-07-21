# Model Card — Flight Delay Risk v1.5.0

## Self-Explaining Product UI Release

Flight Delay Risk estimates the calibrated probability that a scheduled U.S. domestic flight will arrive at least 15 minutes late (`ArrDel15 = 1`) using only information available before departure.

The committed artifact is an Extra Trees pipeline selected by PR-AUC in a dedicated chronological development experiment, then frozen and refitted at larger scale. The release compares seven deliberately selected model candidates, uses 112 features in six auditable families, selects calibration on a later holdout and evaluates the scaled artifact once on an untouched final period.

## Intended use

- rank scheduled flights for human review;
- demonstrate leakage-safe temporal ML engineering;
- expose probability, cohort support and model-behaviour evidence;
- support a bilingual portfolio demo through Streamlit, FastAPI and PDF reports.

It is not intended for dispatch, compensation, passenger guarantees or automated operational decisions.

## Data

- Source: U.S. DOT/BTS Reporting Carrier On-Time Performance, 2024.
- Canonical cleaned rows: **6,965,267**.
- Calendar coverage: January 1–December 31, 2024.
- Development experiment: deterministic proportional sample of 30,000 rows used for model-family selection, ablation and temporal comparison.
- Deployment refit: separate deterministic proportional sample of 250,000 rows, used only after Extra Trees had been frozen.
- Target-free schedule context: fitted on all **6,965,267** published schedules.

### Frozen model-selection experiment

| Partition | Period | Rows | Role |
|---|---|---:|---|
| Model training | 2024-01-01 → 2024-07-16 | 16,045 | Fit all candidates |
| Model selection | 2024-07-17 → 2024-09-04 | 4,191 | Select family by PR-AUC |
| Calibration | 2024-09-05 → 2024-10-18 | 3,701 | Select/refit calibrator and threshold |
| Development test | 2024-10-19 → 2024-12-31 | 6,063 | Diagnose temporal transfer before scale refit |

### Deployed scaled artifact

| Partition | Period | Rows | Role |
|---|---|---:|---|
| Frozen-finalist refit | 2024-01-01 → 2024-09-04 | 168,519 | Refit Extra Trees without model-family reselection |
| Calibration | 2024-09-05 → 2024-10-18 | 31,028 | Select and refit probability calibrator |
| Untouched final test | 2024-10-19 → 2024-12-31 | 50,453 | Reporting-only evaluation of the deployed artifact |

## Feature system

The artifact uses 112 features:

| Family | Count | Description |
|---|---:|---|
| Core schedule | 32 | Carrier, route, scheduled times, duration, distance and deterministic schedule transforms |
| Calendar | 16 | Exact date position, season, annual cycles and federal-holiday proximity |
| Historical rates | 16 | Smoothed long-run target rates and frequency shares |
| Historical support | 20 | Exact and log cohort counts |
| Recency | 16 | 28-day, 90-day, EWMA and short-vs-long trend features |
| Scheduled congestion | 12 | Target-free timetable density and bank concentration |

Training-row target-derived features use outcomes from strictly earlier dates. Same-day rows are transformed before that date updates state. Scheduled-congestion context contains no target values and is shared between training and serving.

## Candidate benchmark

The public scope compares a linear baseline, Random Forest, Extra Trees, XGBoost, LightGBM, an embedding MLP and FT-Transformer on the same chronological selection block.

| Candidate | ROC-AUC | PR-AUC | Lift@10% |
|---|---:|---:|---:|
| **Extra Trees** | **0.6685** | **0.3728** | **1.784×** |
| Random Forest | 0.6633 | 0.3637 | 1.744× |
| Logistic Regression | 0.6486 | 0.3586 | 1.774× |
| LightGBM | 0.6573 | 0.3577 | 1.656× |
| XGBoost | 0.6566 | 0.3524 | 1.665× |
| MLP with embeddings | 0.6481 | 0.3442 | 1.656× |
| FT-Transformer | 0.6416 | 0.3330 | 1.439× |

Extra Trees was selected under the declared PR-AUC rule. Elastic Net, sklearn HistGradientBoosting and CatBoost are no longer active public candidates because they added comparison volume without improving the portfolio narrative.

### Neural diagnostics

| Candidate | Parameters | Epochs | Best inner validation loss |
|---|---:|---:|---:|
| MLP with embeddings | 302,010 | 6 | 1.1111 |
| FT-Transformer | 132,457 | 7 | 1.1047 |

Both use chronological inner validation, early stopping and joblib serialization.

## Feature ablation

Drop-one-family ablation uses the same Extra Trees configuration and chronological model-training/selection blocks.

- Full 112 features: PR-AUC `0.3728`, Lift@10% `1.784×`.
- Core only: PR-AUC `0.3602`.
- Removing calendar: Δ PR-AUC `-0.0061`.
- Removing historical rates: Δ PR-AUC `-0.0041`.
- Removing support: Δ PR-AUC `-0.0048`, but Δ Lift `+0.020×`.
- Removing recency: Δ PR-AUC `-0.0023`, but Δ Lift `+0.049×`.
- Removing schedule congestion: Δ PR-AUC `-0.0038`.

The full system wins the declared selection metric, but not every family improves every ranking statistic.

## Development calibration experiment

Candidate calibrators are fitted on September 5–26 and selected on September 27–October 18. Isotonic achieved the best later-holdout Brier score (`0.1236`) in the 30,000-row development experiment and was refitted on its 3,701 calibration rows. This documents the protocol; it is not the calibrator of the final scaled artifact.

| Metric | Raw | Calibrated |
|---|---:|---:|
| Brier score | 0.2266 | 0.1409 |
| Log loss | 0.6448 | 0.4576 |
| ECE | 0.2922 | 0.0304 |

## Scaled-artifact calibration

The same chronological method-selection protocol was rerun after the frozen Extra Trees finalist was refitted at scale. Sigmoid narrowly beat isotonic on the later calibration holdout (`0.120531` versus `0.120560` Brier) and was refitted on all 31,028 calibration rows.

## Untouched scaled-artifact performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6179 |
| PR-AUC | 0.2386 |
| F1 | 0.2460 |
| Precision@Top10% | 0.2801 |
| Lift@Top10% | 1.6387× |
| Brier score | 0.1385 |
| ECE | 0.0130 |

The scaled refit improved the reported final-period metrics relative to the smaller development artifact. Because the samples differ, those deltas are not treated as paired row-level estimates. The final test was not reused to change the frozen model family, feature set or policy.

## Temporal backtest

Across three expanding folds with the seven public candidates:

- MLP with embeddings selected in 1/3 folds;
- FT-Transformer selected in 1/3 folds;
- Extra Trees selected in 1/3 folds;
- sigmoid calibration selected in 2/3 folds;
- isotonic calibration selected in 1/3 folds;
- mean PR-AUC: `0.2835`;
- mean Lift@10%: `1.717×`;
- mean Brier score: `0.1523`;
- mean ECE: `0.0379`.

No model family is stable across all periods.

## Explanation

For Extra Trees, decision paths are decomposed into parent-to-child positive-class probability changes, averaged across trees, grouped to raw features and rescaled to the pre-calibration log-odds change. This explains model behaviour, not causes.

## Limitations

- No live weather, ATC, aircraft rotation, crew, gate or maintenance state.
- One calendar year and a release-size training sample.
- Scheduled congestion is a timetable proxy, not observed congestion.
- Cohort signals are associative rather than causal.
- The F1 threshold is not an operational cost/capacity policy.
- Calibration and model-family performance can drift over time.


## Operational robustness

The deployed decision surface uses an exact top-10% review-capacity policy. Weekly block-bootstrap confidence intervals are reported for ranking and calibration metrics, and paired intervals compare Extra Trees with the calibrated Logistic baseline. The model family was frozen before this analysis. Feature drift is high in the October–December test block, so the policy requires monitoring and periodic recalibration.
