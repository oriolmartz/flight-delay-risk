# Weather ablation protocol

The weather experiment compares three scopes on one identical chronological cohort:

1. `baseline_current`: the current complete FlightRisk feature schema.
2. `baseline_plus_weather`: current features plus point-in-time origin/destination weather.
3. `weather_only`: compact schedule/calendar context plus weather.

Rows are retained only when weather is available for both endpoints. This makes the
incremental comparison fair and prevents missingness from becoming the experimental
treatment. The result is explicitly cohort-specific, not a global-network claim.

Weather observations obey `observation_time_utc <= prediction_cutoff_utc`, where the
cutoff is T-6 hours by default. Numeric weather gaps are median-imputed using the model
training block only; future selection rows never influence imputation.

Run:

```bash
python -m scripts.run_weather_ablation \
  --data data/processed/flights_with_weather_2024.parquet \
  --max-rows 300000 \
  --candidate extra_trees
```

Outputs:

- `reports/weather_ablation.json`
- `reports/weather_ablation.md`
