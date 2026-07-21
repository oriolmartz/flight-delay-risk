# Flight Delay Risk limitations

Flight Delay Risk is a portfolio-grade pre-departure risk-ranking system, not an operational airline decision engine.

## Data and scope

- The canonical dataset covers U.S. domestic BTS records for one calendar year, 2024.
- Model-family selection and feature ablation use a deterministic 30,000-row proportional sample; they are controlled development experiments, not a full-data hyperparameter search.
- After freezing Extra Trees, the deployed release was refitted from a separate deterministic 250,000-row sample: 168,519 refit rows, 31,028 calibration rows and 50,453 untouched test rows.
- Cancelled, diverted and target-missing flights are excluded from the supervised target population.
- Transfer to European operations is not validated or calibrated.

## Missing live operational state

The model does not receive live weather, ATC restrictions, aircraft tail rotation, inbound delay propagation, crew legality, gate availability, maintenance state or real-time airport operations. Scheduled-congestion features are a timetable-density proxy, not observed congestion.

## Non-stationarity

Feature families improved PR-AUC on the July–September selection block, but the selected v1.3.0 model achieved lower ranking metrics on the October–December untouched test than the prior feature release. This is evidence of temporal drift and feature-period interaction, not a claim of monotonic model improvement.

## Feature ablation boundaries

- Ablation conclusions are specific to Extra Trees, the fixed release configuration and the declared selection period.
- A family can improve PR-AUC while reducing point Lift@10%, or vice versa.
- Correlated families can mask one another; drop-one ablation is not causal attribution.
- Historical rates, support and recency are associative and can reflect structural differences between routes, carriers and airports.

## Probability and threshold

Calibration improves probability quality on the held-out period but can drift when base rates or operations change. The F1-derived release threshold is an experimental default, not an operational capacity or cost policy.

## Explanation

Local contributions describe how the selected estimator moved its pre-calibration log-odds. They do not establish why a flight will be delayed or identify an intervention.

## Production gaps

A real deployment would require outcome joins, calibration and performance monitoring, data-quality alerts, secure persistence, authentication, service-level objectives, retraining policy and governance review.
