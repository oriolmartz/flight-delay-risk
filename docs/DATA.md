# Flight Delay Risk data guide

## Primary source

Flight Delay Risk uses the **U.S. DOT / Bureau of Transportation Statistics (BTS) TranStats — Reporting Carrier On-Time Performance** dataset.

The supervised target is:

```text
ArrDel15 = 1 if the flight arrived 15+ minutes late, otherwise 0
```

Raw monthly CSVs belong under `data/raw/` and remain ignored by Git.

## Canonical 2024 dataset

The current canonical parquet contains:

- **7,079,081** source rows read from 12 monthly files;
- **113,814** cancelled or diverted rows excluded;
- **6,965,267** supervised rows written;
- complete coverage from **2024-01-01 through 2024-12-31**;
- all **366 calendar days** represented;
- target positive rate of approximately **20.82%**.

The auditable manifest is written to:

```text
data/processed/data_manifest.json
```

It records:

- SHA-256 for every raw source;
- selected/non-selected duplicate-month files;
- preparation configuration and its fingerprint;
- row-level cleaning totals;
- processed dataset SHA-256;
- schema and dtypes;
- date range and monthly calendar coverage.


## Target-free schedule context

Layer 3 builds a reusable timetable-density reference from the complete canonical parquet:

```bash
python -m scripts.build_schedule_context
```

The resulting `data/processed/schedule_context.joblib` contains no delay labels. It stores expected departure/arrival density by weekday, airport and schedule slot, plus daily carrier-route volumes. The same cache is loaded by training, backtesting, ablation and serving so congestion features do not depend on the release sample and do not drift between train and inference code.

The artifact metadata records the context source row count, date range and scope. The release manifest fingerprints the cache.

## Prepare the data

```bash
python -m scripts.prepare_data \
  --input-dir data/raw \
  --output data/processed/flights_processed.parquet \
  --manifest data/processed/data_manifest.json
```

Preparation is chunked, so the full year does not need to fit in memory.

## Duplicate-month protection

Two files claiming the same `(Year, Month)` are rejected by default before the full dataset is read:

```bash
python -m scripts.prepare_data --duplicate-month-policy error
```

Explicit alternatives exist for recovery workflows:

```text
prefer-largest
prefer-newest
first
```

They should only be used deliberately. The manifest preserves which file was selected and why.

## Sampling without head bias

A development sample must represent the complete source file. Flight Delay Risk therefore uses a deterministic uniform sample over each monthly CSV instead of `nrows`, which would retain only the first days of every month.

```bash
python -m scripts.prepare_data --sample-rows-per-file 50000
```

The training command also supports deterministic proportional sampling across all observed months:

```bash
python -m scripts.train_model --max-rows 30000 --candidate-profile flagship
```

The sampled frame is sorted chronologically before partitioning.

## Download options

### Manual TranStats download

Select:

- Database: On-Time;
- Table: Reporting Carrier On-Time Performance;
- Year and month;
- CSV/prezipped export.

Place one file per month in `data/raw/`.

### Best-effort helper

```bash
python -m scripts.run_real_data_demo --download --year 2024 --months 1 2 3
```

TranStats download URLs can be session-generated, so manual download remains the reliable fallback.

## Required fields

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

Official uppercase/snake-case names such as `FL_DATE`, `OP_UNIQUE_CARRIER`, `CRS_DEP_TIME` and `ARR_DEL15` are normalized automatically.

## Validation and cleaning

The cleaner:

- parses `FlightDate` during ingestion;
- validates binary target values;
- filters cancelled/diverted rows before supervised modeling;
- validates numeric ranges and HHMM schedule fields;
- verifies year, month and day-of-week consistency against `FlightDate`;
- rejects missing carrier/origin/destination identifiers;
- drops post-flight leakage columns;
- records exact duplicate observations for audit.

## Leakage columns that must not be features

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

`Cancelled` and `Diverted` are filtering fields only.

## Processed files

Preferred output:

```text
data/processed/flights_processed.parquet
```

CSV output is supported for small workflows, but the canonical full-year build uses chunked Parquet with Zstandard compression.
