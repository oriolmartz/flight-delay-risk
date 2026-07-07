# Model evaluation

FlightRisk evaluates two related but different modeling paths.

## U.S. BTS flight-level model

The U.S. model predicts whether a scheduled flight will arrive 15+ minutes late (`ArrDel15`).

Primary metrics:

- **ROC-AUC**: broad ranking quality across thresholds.
- **PR-AUC**: more useful than accuracy when the positive class is relatively sparse.
- **F1**: thresholded classification balance.
- **Precision@Top10%**: observed delay rate among the highest-risk decile.
- **Lift@Top10%**: how much better the top-risk decile is than random selection.

For a flight-risk product, Lift@10 is especially useful: it answers whether the model concentrates real delays in the set of flights it marks as highest risk.

## Temporal validation

The main training script uses:

```text
train/validation/test split ordered by FlightDate
```

v6.5 adds:

```bash
python -m scripts.run_temporal_backtest --n-splits 3
```

This performs expanding-window evaluation:

```text
fold 1: train early period -> test next period
fold 2: train larger early period -> test next period
fold 3: train even larger early period -> test final period
```

## Confidence intervals

v6.5 adds bootstrap intervals for the final held-out evaluation:

```bash
python -m scripts.run_real_data_demo --bootstrap-samples 200
```

The report is saved in:

```text
reports/metrics.json
```

Look for:

```json
"confidence_intervals": {
  "roc_auc": {"lower": ..., "upper": ...},
  "pr_auc": {"lower": ..., "upper": ...},
  "f1": {"lower": ..., "upper": ...}
}
```

## Hyperparameter search

v6.5 adds a modest time-aware search:

```bash
python -m scripts.tune_hyperparameters --model logistic_regression --n-iter 12
python -m scripts.tune_hyperparameters --model extra_trees --n-iter 12
```

Results are saved in:

```text
reports/hyperparameter_search.json
```

The main pipeline remains fixed by default for reproducibility. The tuning script is meant to show how the fixed candidates can be improved.
