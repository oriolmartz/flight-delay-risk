# Flight Delay Risk v1.5.0 — Scale refit and deployment readiness

## Release decision

The scaled refit release freezes the Extra Trees model family and the exact top-10% review policy selected in previous layers. It does **not** reopen model-family selection on the final quarter. The goal is to test whether the chosen system survives a materially larger refit and whether the packaged application satisfies a public deployment contract.

## Scaled chronological build

The deterministic release sample increases from 30,000 to **250,000 flights** across all twelve months of 2024, an 8.33× increase.

| Block | Dates | Rows |
|---|---|---:|
| Model training | 2024-01-01 → 2024-07-16 | 133,599 |
| Prior selection, inherited into refit | 2024-07-17 → 2024-09-04 | 34,920 |
| Frozen-finalist refit | 2024-01-01 → 2024-09-04 | **168,519** |
| Calibration | 2024-09-05 → 2024-10-18 | 31,028 |
| Untouched test | 2024-10-19 → 2024-12-31 | **50,453** |

Historical target encodings remain strictly prior-date. The complete target-free schedule context still uses all 6,965,267 cleaned schedules.

## Scale engineering

The prior one-hot Extra Trees path was not memory-efficient at this size. v1.4 keeps the estimator family frozen but moves it to a compact `float32` ordinal representation. The comparison baseline is explicitly renamed **SGD numeric logistic baseline**; it is not presented as the exact v1.3 logistic estimator.

Fitting is checkpointed in an isolated process so native worker pools cannot contaminate calibration, testing or serving. The fit payload contains only the fitted pipelines, historical aggregates, drift reference and split lineage.

Measured build characteristics:

- Extra Trees fit: 7.75 s.
- SGD logistic fit: 1.23 s.
- End-to-end release build: 46.42 s, excluding the external checkpoint handoff ambiguity.
- Peak resident memory: 1.31 GB.
- Packaged artifact: 52.4 MB.

A 500,000-row experiment was attempted, but the recency-aware historical encoder exceeded the practical build budget before model fitting. The release therefore stops at 250,000 rather than hiding a non-reproducible scale claim.

## Untouched-test result

| Metric | v1.5.0 |
|---|---:|
| ROC-AUC | 0.6179 |
| PR-AUC | 0.2386 |
| Precision@10% | 0.2801 |
| Lift@10% | 1.6387× |
| Brier score | 0.1385 |
| Expected calibration error | 0.0130 |

The exact 10% policy reviews 5,046 of 50,453 flights, with precision 0.2800, recall 0.1639 and lift 1.6384×.

Weekly block bootstrap uses 100 repetitions in this larger evaluation. The smaller v1.3 release retains the more expensive 500-repetition audit. v1.4 prioritizes a larger untouched test and reports the reduced bootstrap budget explicitly.

## Deployment contract

The release exposes:

- `GET /live`: process liveness only;
- `GET /ready`: artifact, schedule-context, report, version and schema readiness;
- `GET /health`: backwards-compatible application health;
- `GET /openapi.json`: versioned API contract;
- `x-request-id`, `x-flightrisk-version` and `x-process-time-ms` response headers.

Docker, Compose and Render health checks use readiness rather than a superficial process check. Runtime paths can be overridden with environment variables.

The committed production smoke exercises liveness, readiness, model metadata, prediction, ranking and OpenAPI through `TestClient`. It is a packaging verification, not evidence of external uptime.
