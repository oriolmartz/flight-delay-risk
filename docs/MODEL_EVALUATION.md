# Flight Delay Risk model evaluation

## Protocol

Flight Delay Risk separates four complete-date blocks:

```text
model_train: 2024-01-01 → 2024-07-16   16,045 rows
selection:   2024-07-17 → 2024-09-04    4,191 rows
calibration: 2024-09-05 → 2024-10-18    3,701 rows
test:        2024-10-19 → 2024-12-31    6,063 rows
```

1. Fit the complete target-free timetable context independently of labels.
2. Build long-run, support and recency features using strictly prior dates.
3. Fit all seven public candidate families on `model_train`.
4. Select the family by PR-AUC on `selection`.
5. Refit the winner on `model_train + selection`.
6. Fit calibration candidates on the first calibration half.
7. Select calibration method on the later calibration half.
8. Refit the calibrator and tune threshold on the complete calibration block.
9. Evaluate once on `test`.

## Selection benchmark

Extra Trees won selection PR-AUC (`0.3728`). Random Forest reached `0.3637`, Logistic Regression `0.3586`, LightGBM `0.3577`, XGBoost `0.3524`, MLP embeddings `0.3442` and FT-Transformer `0.3330`.

## Ablation

The full 112-feature system produced PR-AUC `0.3728`. Core-only reached `0.3602`. Every added family improved selection PR-AUC in the drop-one comparison, with marginal contributions ranging from `0.0023` for recency to `0.0061` for calendar. Point Lift@10% did not move monotonically: removing support or recency increased lift while reducing PR-AUC.

## Calibration

Isotonic was selected on the later chronological calibration holdout:

| Method | Brier | Log loss | ECE |
|---|---:|---:|---:|
| Identity | 0.2207 | 0.6326 | 0.3099 |
| Sigmoid | 0.1239 | 0.4086 | 0.0191 |
| **Isotonic** | **0.1236** | **0.4074** | **0.0110** |

On the untouched test, calibration reduced Brier from `0.2266` to `0.1409` and ECE from `0.2922` to `0.0304`.

## Untouched test

| Metric | Extra Trees |
|---|---:|
| ROC-AUC | 0.5849 |
| PR-AUC | 0.2073 |
| Precision@Top10% | 0.2327 |
| Lift@Top10% | 1.363× |
| Brier | 0.1409 |
| ECE | 0.0304 |

The selection-to-test drop is treated as a non-stationarity finding. The test is not reused to remove feature families or choose another candidate.

## Expanding backtest

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.6163 | 0.0537 | 0.5821 | 0.6783 |
| PR-AUC | 0.2835 | 0.1137 | 0.1942 | 0.4115 |
| Lift@10% | 1.717× | 0.1321 | 1.577× | 1.840× |
| Brier | 0.1523 | 0.0237 | 0.1331 | 0.1787 |
| ECE | 0.0379 | 0.0310 | 0.0135 | 0.0727 |

Selected families: MLP 1, FT-Transformer 1, Extra Trees 1. Selected calibration methods: sigmoid 2, isotonic 1.

## Interpretation

The strongest evidence in v1.5.0 is not a universal winning model. It is the reproducible demonstration that feature value, model family and calibration method vary across time even under a fixed leakage-safe protocol.


## Operational robustness

The deployed decision surface uses an exact top-10% review-capacity policy. Weekly block-bootstrap confidence intervals are reported for ranking and calibration metrics, and paired intervals compare Extra Trees with the calibrated Logistic baseline. The model family was frozen before this analysis. Feature drift is high in the October–December test block, so the policy requires monitoring and periodic recalibration.
