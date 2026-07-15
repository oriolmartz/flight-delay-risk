# Flight Delay Risk feature ablation

Release: **v1.3.0**

Each row retrains the same model on the same chronological blocks. Only the named feature family changes.

Candidate: `extra_trees` · selection metric: `pr_auc`

| Scope | Features | PR-AUC | Δ PR-AUC | Lift@10% | Δ Lift | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| `full` | 112 | 0.3728 | +0.0000 | 1.784× | +0.000× | 0.6685 |
| `without_core_schedule` | 80 | 0.3385 | -0.0342 | 1.606× | -0.177× | 0.6327 |
| `without_calendar` | 96 | 0.3667 | -0.0061 | 1.764× | -0.020× | 0.6669 |
| `without_historical_rates` | 96 | 0.3686 | -0.0041 | 1.705× | -0.079× | 0.6663 |
| `without_historical_support` | 92 | 0.3679 | -0.0048 | 1.803× | +0.020× | 0.6681 |
| `without_recency` | 96 | 0.3704 | -0.0023 | 1.833× | +0.049× | 0.6679 |
| `without_schedule_congestion` | 100 | 0.3689 | -0.0038 | 1.754× | -0.030× | 0.6676 |
| `core_only` | 32 | 0.3602 | -0.0125 | 1.725× | -0.059× | 0.6629 |
| `core_plus_historical` | 48 | 0.3600 | -0.0127 | 1.734× | -0.049× | 0.6661 |

## Interpretation guardrail

A performance fall after removing a family means that family helped on this selection period. A performance rise means it did not generalise in this run; the negative result is retained rather than reclassified as a win.

> **Layer 4 guardrail:** Core model-selection evidence was carried forward from v1.2.0. The model family remained frozen while policy, uncertainty and drift were evaluated.
