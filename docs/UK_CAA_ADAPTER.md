# UK CAA Adapter

FlightRisk v5.3 adds an official-data-ready UK CAA punctuality adapter.

## Why v5.3 instead of v6?

A full v6 would mean a European flight-level dataset comparable to BTS: scheduled time, actual arrival, route, carrier, airport, delay target and leakage-safe fields across many flights. That is a larger data-access and normalization project.

v5.3 is the realistic portfolio step:

```text
official UK CAA punctuality CSVs
-> adapter
-> canonical European context schema
-> European route intelligence in UI/API
```

The core ML model remains BTS-trained; the European layer becomes official-data-ready and can use real aggregate context as soon as the CAA CSVs are placed in the project.

## Download flow

1. Go to the UK CAA flight punctuality statistics pages.
2. Download monthly CSV resources such as:
   - Punctuality Statistics Full Analysis
   - Punctuality Statistics Full Analysis Arrival Departure
   - Summary Analysis CSV
3. Put them in:

```text
data/europe/uk_caa_raw/
```

4. Run:

```bash
python -m scripts.prepare_uk_caa_context
```

5. The script writes:

```text
data/europe/europe_punctuality_context.csv
```

6. Restart FastAPI and Streamlit.

## Canonical schema

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

## Important honesty note

If `data/europe/europe_punctuality_context.csv` exists, FlightRisk uses it.

If it does not exist, FlightRisk falls back to:

```text
data/europe/europe_punctuality_sample.csv
```

That fallback is a demo/test sample, not official data.
