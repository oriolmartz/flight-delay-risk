# FlightRisk v6.1 Predictive Upgrade

v6.1 focuses on improving the **USA BTS predictive layer** without adding leakage.

The previous USA model was useful as an ML engineering demo, but its PR-AUC/F1 were modest because it only used basic schedule fields and coarse historical aggregates. v6.1 adds stronger pre-flight features and ranking/product metrics.

## New leakage-safe features

All features are known before departure or computed from the training split only.

### Schedule features

- `DepPeriod`, `ArrPeriod`
- `DistanceBand`
- `IsMorningPeak`, `IsEveningPeak`, `IsPeakHour`
- `IsRedEye`
- `IsLongHaul`
- `ScheduledSpeedMph`
- `LogDistance`
- cyclic hour features:
  - `DepHourSin`, `DepHourCos`
  - `ArrHourSin`, `ArrHourCos`

### Historical aggregate features

Computed on the training split only:

- `CarrierRouteDelayRate`
- `AirlineOriginDelayRate`
- `AirlineDestDelayRate`
- `OriginHourDelayRate`
- `DestHourDelayRate`
- `CarrierDepHourDelayRate`

### Schedule-density features

Also computed on the training split only:

- `RouteFlightShare`
- `CarrierRouteFlightShare`
- `AirlineOriginFlightShare`
- `OriginHourFlightShare`
- `DestHourFlightShare`
- `CarrierDepHourFlightShare`

## Model candidates

Default candidates:

- Logistic Regression baseline
- L1 Logistic Regression
- Random Forest
- Extra Trees

Dense `GradientBoostingClassifier` remains opt-in with `--include-gradient-boosting` because it can be very slow on BTS one-hot features.

## New metrics

v6.1 adds ranking metrics to `reports/metrics.json`:

- `precision_at_top_5pct`
- `precision_at_top_10pct`
- `precision_at_top_20pct`
- `recall_at_top_10pct`
- `lift_at_top_10pct`
- `baseline_positive_rate`

These are more product-relevant than F1 for a delay-risk system. The key question becomes:

> Among the flights ranked highest-risk by the model, how much more delayed are they than the average flight?

## Recommended command

```bash
python -m scripts.run_real_data_demo --selection-metric pr_auc
```

For faster iteration:

```bash
python -m scripts.run_real_data_demo --max-rows 200000 --selection-metric pr_auc
```
