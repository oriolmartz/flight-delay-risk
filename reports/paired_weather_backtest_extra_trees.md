# Paired ExtraTrees weather backtest

Release: **v1.5.0**

This experiment freezes the ExtraTrees model and compares the current baseline against the same model plus point-in-time weather. Every fold uses identical rows, chronological windows, hyperparameters and random seed.

Complete-weather cohort: **1,406,680 flights**

| Fold | Train period | Test period | Base PR-AUC | Weather PR-AUC | ΔPR-AUC | Base Lift@10% | Weather Lift@10% | ΔLift | Base Brier | Weather Brier | ΔBrier |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2024-01-01 → 2024-07-01 | 2024-07-02 → 2024-08-31 | 0.4612 | 0.4646 | +0.0034 | 1.793× | 1.815× | +0.022× | 0.2439 | 0.2440 | +0.0000 |
| 2 | 2024-01-01 → 2024-08-31 | 2024-09-01 → 2024-10-31 | 0.2446 | 0.2472 | +0.0026 | 1.859× | 1.904× | +0.045× | 0.2415 | 0.2378 | -0.0036 |
| 3 | 2024-01-01 → 2024-10-31 | 2024-11-01 → 2024-12-31 | 0.2656 | 0.2744 | +0.0087 | 1.667× | 1.742× | +0.075× | 0.1778 | 0.1769 | -0.0010 |

## Aggregate paired deltas

- Mean ΔPR-AUC: **+0.0049**
- Mean ΔLift@10%: **+0.047×**
- Mean ΔBrier: **-0.0015** (lower is better)
- Weather PR-AUC wins: **3/3 folds**
- Weather Lift@10% wins: **3/3 folds**

## Interpretation guardrail

The estimate applies to flights with valid point-in-time weather at both endpoints. It isolates incremental weather value; it is not a new model-selection tournament.