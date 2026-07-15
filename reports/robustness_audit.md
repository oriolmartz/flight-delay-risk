# Scaled robustness audit

Bootstrap samples: **100** complete-week resamples.

| Metric | Point | 95% lower | 95% upper |
|---|---:|---:|---:|
| `roc_auc` | 0.6179 | 0.6038 | 0.6276 |
| `pr_auc` | 0.2386 | 0.2036 | 0.2813 |
| `brier_score` | 0.1385 | 0.1216 | 0.1558 |
| `expected_calibration_error` | 0.0130 | 0.0044 | 0.0390 |
| `lift_at_top_10pct` | 1.6387 | 1.5096 | 1.7510 |