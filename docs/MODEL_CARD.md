# Model Card — FlightRisk

## Intended use

FlightRisk predicts the probability that a **scheduled U.S. domestic
flight** will arrive **15 or more minutes late** (`ArrDel15 = 1`), using
only information available before the flight departs.

**Intended for:**

* Portfolio / educational demonstration of an end-to-end ML system
  (data engineering, feature engineering, leakage-safe modeling,
  evaluation, API serving, and a demo dashboard).
* Illustrating general historical patterns in U.S. domestic flight delays
  (e.g. evening flights and certain routes/carriers tend to run later on
  average).

## Non-intended use

* **Not for operational aviation, air-traffic, or safety decisions.**
* **Not for real travel booking or itinerary decisions** -- do not use this
  to decide whether to book a flight, buy travel insurance, or plan a
  connection.
* Not validated for fairness/bias across carriers or regions beyond basic
  historical rate features; do not use for any decision that could
  disadvantage a carrier, airport, or group of travelers.
* Not a replacement for live, carrier-provided flight status information.

## Dataset

* Source: U.S. DOT / Bureau of Transportation Statistics (BTS) TranStats,
  *Reporting Carrier On-Time Performance* dataset. See `docs/DATA.md`.
* The bundled demo uses a small **synthetic** sample dataset
  (`data/sample/sample_flights.csv`) with realistic but artificially
  generated delay patterns; real evaluation results depend entirely on
  which BTS months/years the user downloads and trains on.

## Features

All features are derived from information known **before** a flight
departs:

* Scheduled departure/arrival hour (derived from `CRSDepTime`/`CRSArrTime`)
* Month, day of week, weekend flag
* Scheduled elapsed time, distance
* Carrier, origin airport, destination airport, route identifier
  (`Origin_Dest`)
* Historical delay rates for carrier / route / origin / destination,
  computed **only from the training split** (see "Leakage controls" below)

See `src/config.py::FEATURE_COLUMNS` for the authoritative list.

## Target

`ArrDel15`: binary indicator, 1 if the flight's arrival delay is 15 minutes
or more, 0 otherwise. Rows with a missing target, and cancelled/diverted
flights (which have no meaningful arrival-delay outcome), are excluded from
training and evaluation.

## Metrics

Reported for both the baseline (Logistic Regression) and the selected candidate model on a held-out test split. v3 selects among Random Forest and Extra Trees candidates using validation PR-AUC by default, then tunes the decision threshold on the validation split before final test reporting:

* ROC-AUC, PR-AUC (Average Precision)
* Precision, Recall, F1 at the tuned decision threshold stored in the model artifact
* Confusion matrix, full classification report
* Calibration curve data (predicted probability vs. observed frequency)
* Feature importance where supported by the selected model

See `reports/metrics.json` after running training for actual numbers on
your data -- results depend heavily on which months/years of BTS data are
used and are not hardcoded here.

## Limitations

* Flight delays are influenced by many factors not present in this dataset
  (real-time weather, air-traffic control conditions, aircraft rotation
  issues, crew scheduling) -- the model captures **statistical, historical**
  patterns only, not real-time causes.
* Historical delay-rate features can encode indirect proxies for
  seasonal/regional effects; interpret "top factors" as descriptive
  associations, not causal explanations.
* Small or sparse historical data for a given carrier/route/airport falls
  back to a global average rate (see "Leakage controls"), which reduces
  precision for rarely-seen combinations.
* Performance will vary across time periods (e.g. holiday travel surges,
  extreme weather years) not represented in the training window.

## Leakage controls

The model must simulate prediction **before** the flight happens. To
enforce this:

1. `src/config.py::FORBIDDEN_LEAKAGE_COLUMNS` explicitly lists post-flight /
   actual-operation columns (actual times, taxi times, delay-cause
   breakdowns, cancellation status, etc.) that must never be used as
   features.
2. `src/data/clean.py` drops all such columns immediately after using
   `Cancelled`/`Diverted` only to filter rows (never as features).
3. `src/features/build_features.py::assert_no_leakage_columns()` is called
   before every model fit/predict and raises an error if any forbidden
   column is present in the feature set.
4. Historical aggregate (delay-rate) features are fit **only on the
   model-training split** via `HistoricalAggregates.fit()`, then applied to
   validation/test/inference data via `.transform()` -- validation, test and
   future outcomes never influence these statistics.
5. Model selection and threshold tuning happen on validation data; final
   metrics are reported on a held-out test split.
6. `tests/test_features.py::TestLeakageGuard`, `tests/test_thresholding.py`
   and `tests/test_training_pipeline.py` automatically test these guarantees.

## Ethical / safety notes

* This is an educational project trained on public, aggregate,
  non-personal data (no passenger-level or personally identifiable
  information is used or produced).
* Predictions should not be presented to end users as authoritative or
  operational; the API and dashboard both include an explicit
  "educational, not operational aviation advice" disclaimer.
* Historical delay-rate features could reflect systemic patterns tied to
  specific airports/carriers; users extending this project for real-world
  use should evaluate for unintended disparate impact before any
  production use.
