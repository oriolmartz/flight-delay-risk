# FlightRisk data guide

## Primary source

FlightRisk is designed for the **U.S. DOT / Bureau of Transportation Statistics (BTS) TranStats — Reporting Carrier On-Time Performance (1987-present)** dataset.

The target is:

```text
ArrDel15 = 1 if the flight arrived 15+ minutes late, otherwise 0
```

The repo does not commit full BTS monthly CSVs because they are large and change over time. Keep raw data under `data/raw/`, which is ignored by Git.

## Option A — reliable manual download from TranStats

Use BTS TranStats and select:

- Database: On-Time
- Table: Reporting Carrier On-Time Performance (1987-present)
- Year: desired year
- Period: desired month
- Download as CSV / prezipped file

Place downloaded CSV files in:

```text
data/raw/
```

Then run the full real-data pipeline:

```bash
python -m scripts.run_real_data_demo
```

Or run each stage manually:

```bash
python -m scripts.prepare_data --input-dir data/raw --output data/processed/flights_processed.parquet
python -m scripts.train_model --data data/processed/flights_processed.parquet --output models/flightrisk_model.joblib
python -m scripts.evaluate_model --model models/flightrisk_model.joblib --data data/processed/flights_processed.parquet
```

## Option B — best-effort direct download

TranStats monthly downloads are generated through the official web UI; the `/PREZIP/` filename can be session/generated and may return 404. The project includes a best-effort helper, but manual download is the reliable route:

```bash
python -m scripts.run_real_data_demo --download --year 2024 --months 1 2 3 --max-rows-per-month 50000
```

If it fails, the script prints exact manual instructions and exits cleanly.

## Recommended fields

The downloader keeps a compact schema. If downloading manually, include at least:

```text
FlightDate
Year
Month
DayOfWeek
Reporting_Airline or Operating_Airline
Origin
Dest
CRSDepTime
CRSArrTime
CRSElapsedTime
Distance
ArrDel15
Cancelled
Diverted
```


### Real BTS column-name compatibility

Official TranStats CSV exports may use uppercase/snake-case names such as:

```text
FL_DATE
OP_UNIQUE_CARRIER
CRS_DEP_TIME
CRS_ARR_TIME
CRS_ELAPSED_TIME
ARR_DEL15
```

FlightRisk normalizes these names automatically to the internal schema:

```text
FlightDate
Airline
CRSDepTime
CRSArrTime
CRSElapsedTime
ArrDel15
```

So both manually downloaded BTS CSVs and the bundled sample schema are supported.

## Why these fields

These fields are available before or at scheduling time, except `ArrDel15`, which is the training target, and `Cancelled`/`Diverted`, which are used only to filter rows during cleaning and are then dropped.

Allowed feature families:

- Calendar: `Month`, `DayOfWeek`, weekend flag
- Schedule: `CRSDepTime`, `CRSArrTime`, scheduled departure/arrival hour
- Flight plan: `CRSElapsedTime`, `Distance`, route
- Categorical identity: carrier, origin airport, destination airport
- Historical aggregate features computed from the training split only

## Leakage columns that must not be features

The following are post-flight or actual-operation columns and are forbidden as model features:

```text
ArrDelay
ArrDelayMinutes
DepDelay
DepDelayMinutes
ActualElapsedTime
AirTime
TaxiOut
TaxiIn
WheelsOff
WheelsOn
DepTime
ArrTime
CarrierDelay
WeatherDelay
NASDelay
SecurityDelay
LateAircraftDelay
Cancelled
Diverted
CancellationCode
```

`Cancelled` and `Diverted` are allowed only as filtering columns inside `src/data/clean.py`; they are dropped before feature engineering.

## Processed files

Preferred processed format:

```text
data/processed/flights_processed.parquet
```

If a Parquet engine such as `pyarrow` is not installed, the project can write/read a CSV fallback:

```text
data/processed/flights_processed.csv
```

## What not to commit

Do not commit:

```text
data/raw/*.csv
data/raw/*.zip
data/processed/*.parquet
data/processed/*.csv
models/*.joblib
```

Commit only code, docs, tests and lightweight sample/demo assets.
