# Weather UI architecture

The weather layer is optional and leakage-sensitive. The public schedule-only prediction remains the primary score.

## Offline artifacts

`reports/weather_ui_summary.json` stores compact airport and airport-hour aggregates for the map and heatmap. It contains historical delay rate, mean point-in-time weather severity, observation support and a descriptive adverse-minus-clear delay-rate difference. The latter is association, not causal attribution.

`models/flightrisk_model_weather_base.joblib` and `models/flightrisk_model_weather.joblib` are paired frozen Extra Trees artifacts. They use identical complete-weather rows, chronological partitions, random seed and model family. Their probability difference is the UI weather-model delta.

## Single-flight lookup

For a scheduled flight, the service:

1. converts origin local departure time to UTC;
2. subtracts the six-hour prediction horizon;
3. joins the latest eligible NOAA observation at origin and destination;
4. rejects future observations and marks observations older than six hours unavailable;
5. displays raw conditions even when paired model artifacts are absent;
6. computes the paired base-vs-weather delta only when both observations and both artifacts exist.

## Build commands

```powershell
python -m scripts.build_weather_ui_summary `
  --data data/processed/flights_with_weather_2024.parquet

python -m scripts.train_weather_release `
  --data data/processed/flights_with_weather_2024.parquet
```

Then launch the product:

```powershell
python -m streamlit run app/dashboard/streamlit_app.py
```

The API exposes the same layer through `GET /weather/summary` and `POST /predict/weather`.
