# FlightRisk temporal backtest

Release: **v1.0.0 — Temporal Validation**

Each fold trains on earlier dates, selects and calibrates on a later validation block, then evaluates on the next unseen block.

## Aggregate results

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `roc_auc` | 0.6317 | 0.0307 | 0.5912 | 0.6655 |
| `pr_auc` | 0.2659 | 0.0811 | 0.1975 | 0.3798 |
| `f1` | 0.3534 | 0.0796 | 0.2847 | 0.4642 |
| `precision_at_top_10pct` | 0.3116 | 0.0970 | 0.2377 | 0.4471 |
| `lift_at_top_10pct` | 1.6569 | 0.1719 | 1.4010 | 1.7645 |
| `brier_score` | 0.1498 | 0.0234 | 0.1221 | 0.1785 |
| `expected_calibration_error` | 0.0550 | 0.0350 | 0.0128 | 0.0842 |

## Fold evidence

### Fold 1

- Train: 2024-01-01 → 2024-05-06 (76,777 rows)
- Validation: 2024-05-07 → 2024-06-09 (21,985 rows)
- Test: 2024-06-10 → 2024-08-05 (26,482 rows)
- Selected model: `logistic_l1`
- Calibration: `isotonic`
- PR-AUC: 0.3798
- Lift@10%: 1.749×
- Brier score: 0.1785
- ECE: 0.0128

### Fold 2

- Train: 2024-01-01 → 2024-06-09 (98,762 rows)
- Validation: 2024-06-10 → 2024-08-05 (26,482 rows)
- Test: 2024-08-06 → 2024-10-01 (25,955 rows)
- Selected model: `logistic_l1`
- Calibration: `isotonic`
- PR-AUC: 0.2660
- Lift@10%: 1.713×
- Brier score: 0.1544
- ECE: 0.0835

### Fold 3

- Train: 2024-01-01 → 2024-08-02 (120,454 rows)
- Validation: 2024-08-03 → 2024-10-01 (30,745 rows)
- Test: 2024-10-02 → 2024-11-06 (24,315 rows)
- Selected model: `logistic_l1`
- Calibration: `isotonic`
- PR-AUC: 0.1975
- Lift@10%: 1.765×
- Brier score: 0.1221
- ECE: 0.0842

### Fold 4

- Train: 2024-01-01 → 2024-09-04 (139,091 rows)
- Validation: 2024-09-05 → 2024-11-06 (36,423 rows)
- Test: 2024-11-07 → 2024-12-11 (24,486 rows)
- Selected model: `logistic_l1`
- Calibration: `isotonic`
- PR-AUC: 0.2204
- Lift@10%: 1.401×
- Brier score: 0.1442
- ECE: 0.0395
