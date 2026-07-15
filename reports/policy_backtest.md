# Operational policy temporal backtest

Release: **v1.3.0**

Frozen model family: `extra_trees` · capacity: 10%

| Fold | Test period | Calibration | Precision | Recall | Lift | Utility/flight |
|---:|---|---|---:|---:|---:|---:|
| 1 | 2024-07-09 → 2024-09-05 | `sigmoid` | 0.4220 | 0.1656 | 1.654× | -0.1794 |
| 2 | 2024-09-06 → 2024-11-03 | `isotonic` | 0.2137 | 0.1466 | 1.466× | -0.1148 |
| 3 | 2024-11-04 → 2024-12-31 | `sigmoid` | 0.2542 | 0.1390 | 1.390× | -0.1432 |

## Aggregate policy stability

- Mean precision: 0.2966
- Mean recall: 0.1504
- Mean lift: 1.503×
- Lift range: 1.390× → 1.654×

Every fold refits the frozen model family and calibration using only earlier dates. The future fold is used once for policy reporting.