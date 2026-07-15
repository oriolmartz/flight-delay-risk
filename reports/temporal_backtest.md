# Flight Delay Risk temporal backtest

Release: **v1.3.0**

Every fold rebuilds target-derived features, selects the model on a later block, calibrates on a separate block and evaluates on the next unseen period.

## Aggregate results

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `roc_auc` | 0.6163 | 0.0537 | 0.5821 | 0.6783 |
| `pr_auc` | 0.2835 | 0.1137 | 0.1942 | 0.4115 |
| `f1` | 0.3573 | 0.1051 | 0.2888 | 0.4784 |
| `precision_at_top_10pct` | 0.3402 | 0.1192 | 0.2614 | 0.4774 |
| `lift_at_top_10pct` | 1.7173 | 0.1321 | 1.5773 | 1.8396 |
| `brier_score` | 0.1523 | 0.0237 | 0.1331 | 0.1787 |
| `expected_calibration_error` | 0.0379 | 0.0310 | 0.0135 | 0.0727 |

## Fold evidence

### Fold 1

- Model train: 2024-01-01 → 2024-05-03 (2,911 rows)
- Selection: 2024-05-04 → 2024-06-03 (778 rows)
- Calibration: 2024-06-04 → 2024-07-01 (735 rows)
- Test: 2024-07-02 → 2024-08-31 (1,549 rows)
- Selected model: `mlp_embeddings`
- Calibration: `sigmoid`
- PR-AUC: 0.4115
- Lift@10%: 1.840×
- Brier score: 0.1787

### Fold 2

- Model train: 2024-01-01 → 2024-06-13 (3,947 rows)
- Selection: 2024-06-14 → 2024-07-25 (1,085 rows)
- Calibration: 2024-07-26 → 2024-08-31 (941 rows)
- Test: 2024-09-01 → 2024-10-31 (1,533 rows)
- Selected model: `ft_transformer`
- Calibration: `isotonic`
- PR-AUC: 0.1942
- Lift@10%: 1.735×
- Brier score: 0.1331

### Fold 3

- Model train: 2024-01-01 → 2024-07-25 (5,032 rows)
- Selection: 2024-07-26 → 2024-09-15 (1,306 rows)
- Calibration: 2024-09-16 → 2024-10-31 (1,168 rows)
- Test: 2024-11-01 → 2024-12-31 (1,494 rows)
- Selected model: `extra_trees`
- Calibration: `sigmoid`
- PR-AUC: 0.2448
- Lift@10%: 1.577×
- Brier score: 0.1451


> **Layer 4 guardrail:** Core model-selection evidence was carried forward from v1.2.0. The model family remained frozen while policy, uncertainty and drift were evaluated.
