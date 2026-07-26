# Weather UI architecture

The weather layer is optional, leakage-sensitive and explicitly historical. The public schedule-only prediction remains the primary score and can be returned for historical or future scheduled flights. The NOAA module is a versioned 2024 replay, not a live forecast feed.

## Offline artifacts

`reports/weather_ui_summary.json` stores compact airport and airport-hour aggregates for the map and heatmap. It contains historical delay rate, mean point-in-time weather severity, observation support and a descriptive adverse-minus-clear delay-rate difference. The latter is association, not causal attribution.

`models/flightrisk_model_weather_base.joblib` and `models/flightrisk_model_weather.joblib` are paired frozen Extra Trees artifacts. They use identical complete-weather rows, chronological partitions, random seed and model family. Their probability difference is the UI weather-model delta.

## Single-flight contract

For every request, the service first returns the official schedule-based prediction. It then classifies the optional weather mode:

- `historical_replay`: exact flight date inside the versioned 2024 replay window;
- `future_schedule_only`: future date, with `weather_available: false`, `weather_delta: null` and `reason: live_forecast_feed_required`;
- `schedule_only_outside_weather_coverage`: historical date outside the replay window;
- `schedule_only`: no exact date supplied.

Only in `historical_replay` mode does the service:

1. convert origin local departure time to UTC;
2. subtract the six-hour prediction horizon;
3. join the latest eligible NOAA observation at origin and destination;
4. reject future observations and mark observations older than six hours unavailable;
5. display reconstructed raw conditions even when paired model artifacts are absent;
6. compute the paired base-vs-weather delta only when both observations and both artifacts exist.

The service never substitutes 2024 observations for a future forecast. Operational future-flight weather would require a live, versioned forecast feed and archived forecast vintages for leakage-safe training.

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

The API exposes the same layer through `GET /weather/summary` and `POST /predict/weather`. The latter always returns the official schedule score and explicit mode fields: `mode`, `operational_for_future_flights`, `requires_live_forecast_feed`, `weather_available`, `weather_delta` and `reason`.
