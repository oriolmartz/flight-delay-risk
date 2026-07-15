# Feature-family stability audit

Release: **v1.3.0**

This audit uses expanding folds inside the pre-calibration period. The final calibration and test outcomes are not consulted.

Fixed candidate: `extra_trees` · permutation repeats: 3

| Family | Mean Δ PR-AUC | Median Δ | Positive folds | Stable |
|---|---:|---:|---:|:---:|
| `core_schedule` | +0.1081 | +0.1069 | 3/3 | yes |
| `calendar` | +0.0024 | +0.0027 | 2/3 | yes |
| `historical_rates` | +0.0018 | +0.0014 | 2/3 | yes |
| `historical_support` | +0.0014 | +0.0015 | 2/3 | yes |
| `recency` | +0.0034 | +0.0033 | 3/3 | yes |
| `schedule_congestion` | +0.0015 | +0.0014 | 3/3 | yes |

## Stable feature policy

Selected families: `core_schedule, calendar, historical_rates, historical_support, recency, schedule_congestion`

Selected features: **112**

A family is marked stable only when its joint permutation reduces PR-AUC in at least two thirds of pre-calibration folds and the median drop is positive.