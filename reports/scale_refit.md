# Scaled finalist refit

Release: **v1.5.0**

Extra Trees was frozen before this layer; the scale baseline uses log-loss SGD for tractable sparse optimization. No model-zoo reselection used the final test period.

## Scale

- Sampled flights: **250,000** of 0 canonical rows
- Finalist refit rows: **168,519**
- Calibration rows: **31,028**
- Untouched test rows: **50,453**
- Scale factor vs v1.3 release sample: **8.3×**
- Peak resident memory: **1310 MB**

## Untouched test

| Metric | v1.3 | v1.4 scaled | Delta |
|---|---:|---:|---:|
| `roc_auc` | 0.5849 | 0.6179 | +0.0330 |
| `pr_auc` | 0.2073 | 0.2386 | +0.0313 |
| `lift_at_top_10pct` | 1.3630 | 1.6387 | +0.2757 |
| `brier_score` | 0.1409 | 0.1385 | -0.0024 |
| `expected_calibration_error` | 0.0304 | 0.0130 | -0.0175 |

The comparison uses the same chronological date boundaries but a much larger deterministic sample, so deltas are evidence about the release pipeline rather than a paired claim on identical rows.