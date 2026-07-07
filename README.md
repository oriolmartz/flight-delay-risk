

## v7.9 Portfolio polish

- Tightened the hero height and reduced the right-side example card footprint.
- Improved the single-flight prediction form with portfolio-style badges and a cleaner card treatment.
- Kept the lighter aviation palette and the v7.6 read-time sampling fix.
# FlightRisk Simple — Delay Probability Model

FlightRisk is a machine-learning project that estimates:

```text
P(ArrDel15 = 1)
```

That means: **the probability that a scheduled flight arrives 15+ minutes late**.

English is the default UI language. A visible selector lets the user switch to Spanish.

## Core idea

The project should be understood in one sentence:

> FlightRisk estimates the probability that a scheduled flight will arrive 15+ minutes late before the flight departs.

## What the main model trains on

- **Dataset:** U.S. BTS On-Time Performance flight-level data.
- **Target:** `ArrDel15`.
- **Positive class:** arrival delay of 15 minutes or more.
- **Inputs:** airline, origin, destination, month, day of week, scheduled departure time, scheduled arrival time, scheduled elapsed time, distance and train-fitted historical aggregate rates.
- **Output:** delay probability, risk level and top explanatory signals.

## Batch mode

Batch mode is secondary. It applies the same delay-probability model to many scheduled flights and sorts them by predicted delay probability.

It does **not** replace the core concept. The core concept is still:

```text
scheduled flight -> probability of 15+ minute arrival delay
```

## Europe context

The Europe layer uses UK CAA aggregate punctuality data as route/airline/month context.

It is experimental and is **not** the core flight-level training dataset. The main trained model is based on U.S. BTS flight-level data.

## Quick start

```powershell
cd flightrisk_simple_delay_probability_en_default
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/dashboard/streamlit_app.py
```

Run tests:

```powershell
pytest
```

## API

Run the API:

```powershell
uvicorn app.api.main:app --reload
```

Main endpoints include:

- `POST /predict` — single flight delay probability.
- `POST /predict/batch` — batch probabilities.
- `POST /rank` — batch probabilities sorted by predicted risk.
- `GET /model/card` — model card.
- `GET /european/context/summary` — UK CAA aggregate context summary.

## Important limitation

This is a portfolio ML project. It is not intended for safety-critical dispatch, legal, compensation or operational aviation decisions.


## v7.7 lighter portfolio UI

- Reworked the visual direction to be lighter and more portfolio-like.
- Reduced the oversized hero headline.
- Added a clean aircraft visual in the hero area.
- Kept the blue aviation identity without the very dark AI/HUD look.
- Preserved English as default and Spanish as selectable UI language.
- Kept the v7.6 read-time sampling fix for quick 12-month experiments.

## v7.6 sampling fix

`--max-rows-per-month` now applies when reading existing CSV files from `data/raw/`, not only during BTS auto-download. This means quick experiments no longer load all 7M+ rows before sampling.

Fast smoke training:

```powershell
python -m scripts.run_real_data_demo --selection-metric pr_auc --bootstrap-samples 0 --max-rows-per-month 5000
```

Expected first log with 12 monthly files: around `Combined 12 files into 60000 total rows with max_rows_per_file=5000`.


## v7.9.1 UI focus

The dashboard is now organized around one simple claim: FlightRisk estimates `P(ArrDel15 = 1)`, the probability that a scheduled flight arrives 15+ minutes late.

The UI includes:

- single-flight prediction as the primary workflow;
- a probability chart after prediction;
- batch mode as a secondary workflow;
- expandable model documentation covering model card, variables, cleaning, candidate models, architecture and training pipeline;
- no Europe context in the main product flow.
