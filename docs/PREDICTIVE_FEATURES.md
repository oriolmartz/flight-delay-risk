# FlightRisk predictive feature system

FlightRisk v1.0.0 uses only schedule-time information and historical evidence available before the prediction date.

## Raw inputs

- airline;
- origin and destination;
- month and day of week;
- scheduled departure and arrival time;
- scheduled elapsed time;
- route distance.

## Derived schedule features

- `Route`, `DepPeriod`, `ArrPeriod`, `DistanceBand`;
- `IsMorningPeak`, `IsEveningPeak`, `IsPeakHour`;
- `IsRedEye`, `IsWeekend`, `IsLongHaul`;
- `ScheduledSpeedMph`, `LogDistance`;
- cyclic departure and arrival hour encodings.

## Historical rate features

- carrier;
- route;
- origin and destination;
- carrier-route;
- airline-origin and airline-destination;
- origin-hour and destination-hour;
- carrier-departure-hour.

## Historical frequency features

- route share;
- carrier-route share;
- airline-origin share;
- origin-hour and destination-hour shares;
- carrier-departure-hour share.

## Ordered training behavior

Target-derived features for a training row use only outcomes from strictly earlier `FlightDate` values. Same-day observations are transformed before their targets enter the state.

This prevents:

- self-target leakage;
- same-day cross-row leakage;
- future-date leakage inside the model-training partition.

## Smoothing and support

Every historical rate uses empirical-Bayes smoothing toward the global model-training rate. The release smoothing strength is 50 rows.

Rate maps and exact count maps are both serialized. Count features are used as product evidence rather than direct classifier inputs, allowing the UI to say whether a route rate is based on 20 or 2,000 observations.

## Unseen categories

Unseen rate keys receive the global smoothed fallback and zero support. Frequency-share keys receive zero. Categorical one-hot features use unknown-category-safe preprocessing.

## Models

The release supports these candidate families:

- Logistic Regression;
- L1 Logistic Regression;
- Random Forest;
- Extra Trees;
- optional Gradient Boosting.

The committed artifact uses L1 Logistic Regression because it was selected in all four expanding temporal folds.

## Product metrics

FlightRisk reports:

- ROC-AUC and PR-AUC;
- Precision, Recall and F1;
- Precision@Top5/10/20%;
- Lift@Top5/10/20%;
- Brier score;
- log loss;
- expected calibration error;
- calibration-curve bins.

Ranking metrics answer whether the model concentrates real delays in the flights it places at the top of the review queue. Calibration metrics answer whether the displayed probabilities are numerically credible.
