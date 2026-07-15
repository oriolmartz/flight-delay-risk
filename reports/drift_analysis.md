# Temporal drift report

Release: **v1.3.0**

Overall feature-drift status: **high**

Calendar, season, recency and cumulative-support variables are expected to shift between January-September reference data and October-December evaluation. Performance monitoring is therefore interpreted alongside feature drift.

## Drift by feature family

| Family | High | Moderate | Low | Maximum value |
|---|---:|---:|---:|---:|
| `core_schedule` | 2 | 0 | 30 | 11.6725 |
| `calendar` | 10 | 0 | 6 | 15.9282 |
| `historical_rates` | 4 | 7 | 5 | 2.8902 |
| `historical_support` | 20 | 0 | 0 | 4.5661 |
| `recency` | 16 | 0 | 0 | 6.9313 |
| `schedule_congestion` | 0 | 0 | 12 | 0.0028 |

## Monthly final-test performance

| Period | Rows | Prevalence | Mean p | PR-AUC | Brier | ECE | Lift@10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2024-10-19 → 2024-10-31 | 1,082 | 0.1266 | 0.1506 | 0.1596 | 0.1111 | 0.0240 | 1.463× |
| 2024-11-01 → 2024-11-30 | 2,463 | 0.1425 | 0.1509 | 0.1781 | 0.1216 | 0.0183 | 1.369× |
| 2024-12-01 → 2024-12-31 | 2,518 | 0.2172 | 0.1600 | 0.2518 | 0.1726 | 0.0575 | 1.279× |