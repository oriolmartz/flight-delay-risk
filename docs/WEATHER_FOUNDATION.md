# Weather foundation: point-in-time contract

This layer adds reproducible NOAA Global Hourly observations without changing the trained model yet.

## Prediction contract

The default prediction horizon is **T-6 hours** relative to scheduled departure. For each origin and destination airport, the pipeline selects the latest weather observation whose timestamp is less than or equal to the prediction cutoff. A forward or nearest-neighbour join is forbidden.

```text
observation_time_utc <= prediction_cutoff_utc
```

Observations older than the configured maximum age are removed and represented through explicit availability and stale flags. Unknown airports degrade to missing weather rather than failing silently.

## Commands

```bash
python -m scripts.download_noaa_weather --year 2024 --airports JFK LAX ORD
python -m scripts.build_weather_dataset \
  --weather-input data/raw/weather/noaa_global_hourly/2024 \
  --flights-input data/processed/flights_clean.parquet \
  --horizon-hours 6 \
  --max-age-hours 6
```

## Outputs

- `data/processed/weather_hourly.parquet`: canonical station observations.
- `data/processed/flights_with_weather.parquet`: flights enriched with origin and destination snapshots.
- Raw measurements, observation timestamps, age, availability, stale flags and derived severity indicators remain separate for auditability.

## Scope boundary

This phase uses historical observations available by the cutoff. It does not claim to reconstruct historical forecast products as issued. The weather family should only be connected to model training after an explicit ablation and temporal validation phase.
