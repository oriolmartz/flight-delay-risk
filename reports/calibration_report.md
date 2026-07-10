# FlightRisk calibration report

Release: **v1.0.0 — Temporal Validation**

The selected `logistic_l1` model is post-calibrated with **isotonic calibration** fitted only on the validation period. The table below is evaluated on the later held-out test period (2024-10-07 → 2024-12-11, 60,481 rows).

| Metric | Raw model score | Calibrated probability | Improvement |
|---|---:|---:|---:|
| Brier score | 0.3036 | 0.1336 | −0.1699 |
| Expected calibration error | 0.3947 | 0.0229 | −0.3718 |
| Log loss | 0.8178 | 0.4378 | −0.3801 |
| Mean prediction | 0.5556 | 0.1786 | observed rate: 0.1609 |

Calibration changes the probability scale, not the ranking order. FlightRisk therefore exposes both `raw_model_score` and `delay_probability` in its API contract.
