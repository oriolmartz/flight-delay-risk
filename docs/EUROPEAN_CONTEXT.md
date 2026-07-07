# European Context Layer

FlightRisk v5.2 adds an experimental European context layer.

The core flight-level model is still trained on U.S. DOT/BTS on-time performance data. The European layer does **not** claim to be a Europe-calibrated flight-level model. It adds:

- curated European airport and airline catalogs,
- route distance estimation from airport coordinates,
- optional aggregated punctuality context by route, airline and month,
- API endpoints for European route inference and context inspection.

## Why this exists

European public aviation data is more fragmented than the BTS flight-level dataset. For a portfolio project, the cleanest engineering approach is:

```text
BTS data -> core flight-level ML model
European context CSV -> route/month punctuality layer
European mode -> transfer-style inference + regional context
```

This keeps the project honest while making it more relevant for European recruiters.

## Context CSV schema

Place European aggregated punctuality data in:

```text
data/europe/europe_punctuality_sample.csv
```

Expected columns:

```text
year
month
airline
origin
destination
airport_pair
avg_arrival_delay_min
pct_flights_15min_late
cancelled_pct
source
```

The repository includes a tiny example CSV for demo/tests. Replace it with official or curated UK CAA / EUROCONTROL-style aggregate exports if available.

## API endpoints

```text
GET  /regions/europe
GET  /regions/europe/context
POST /predict/european
```

`POST /predict/european` returns the core ML prediction plus:

```json
{
  "european_context": {
    "status": "matched",
    "pct_flights_15min_late": 0.31,
    "avg_arrival_delay_min": 14.2,
    "matched_level": "airline_route_month"
  },
  "experimental": true,
  "transfer_note": "European mode combines the BTS-trained flight-level model with an aggregated European punctuality context layer..."
}
```

## Limitations

- Not calibrated on European flight-level data.
- Aggregated context may not exist for every route/month.
- Demo CSV is intentionally small.
- Intended for portfolio demonstration and architecture, not operational aviation decisions.
