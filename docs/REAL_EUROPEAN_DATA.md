# Real European Data Mode (FlightRisk v6)

FlightRisk v6 removes the production sample fallback for European mode. European predictions require a generated context CSV built from real UK CAA punctuality CSVs.

## Commands

```bash
python -m scripts.download_uk_caa_punctuality --year 2024 --kind full-analysis
python -m scripts.prepare_uk_caa_context
python -m scripts.train_european_model
```

Generated files:

```text
data/europe/europe_punctuality_context.csv
models/european_punctuality_model.joblib
```

## Sources

The downloader scrapes the official UK Civil Aviation Authority yearly punctuality statistics page and downloads CSV resources. The CAA page states that punctuality statistics are calculated for selected UK airports and publishes CSV resources such as Full Analysis, Arrival Departure and Summary Analysis files.

## Scope

This is real European **aggregate** punctuality data. It is not equivalent to BTS flight-level data. The system therefore presents:

```text
BTS = flight-level ML baseline/core
UK CAA = real European aggregate punctuality intelligence
```

That is honest and recruiter-defensible.
