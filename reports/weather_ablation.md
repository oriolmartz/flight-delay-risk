# FlightRisk weather ablation

Release: **v1.5.0**

All models use the same chronological partitions and the same complete-weather cohort. 
Weather values are joined at T-6h and imputed from model-training medians only.

Candidate: `extra_trees`

| Scope | Usable / requested | PR-AUC | Δ vs baseline | Lift@10% | Δ vs baseline | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_current` | 112 / 112 | 0.4558 | +0.0000 | 1.848× | +0.000× | 0.6861 | 0.2465 |
| `baseline_plus_weather` | 150 / 150 | 0.4653 | +0.0095 | 1.916× | +0.068× | 0.6895 | 0.2455 |
| `weather_only` | 54 / 54 | 0.4675 | +0.0117 | 1.909× | +0.061× | 0.6893 | 0.2393 |

## Cohort guardrail

This experiment estimates incremental weather value on flights with weather available at both endpoints. It does not yet claim global-network performance.