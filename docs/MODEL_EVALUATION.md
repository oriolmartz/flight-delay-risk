# Model evaluation

FlightRisk evaluates discrimination, ranking and probability quality separately.

## Metrics

### Discrimination and ranking

- ROC-AUC;
- PR-AUC;
- F1 at a validation-selected threshold;
- Precision@TopK and Recall@TopK;
- Lift@TopK against the period-specific positive rate.

### Probability quality

- Brier score;
- log loss;
- expected calibration error;
- quantile reliability curve.

## Main chronological protocol

```text
model training: 2024-01-01 → 2024-08-06
validation:     2024-08-07 → 2024-10-06
test:           2024-10-07 → 2024-12-11
```

1. Fit ordered historical features and candidate pipelines on model-training data.
2. Compare candidates on validation PR-AUC.
3. Compare identity, sigmoid and isotonic calibration on validation Brier score.
4. Tune the classification threshold on calibrated validation probabilities.
5. Freeze the complete artifact.
6. Report once on the held-out test block.

## Candidate benchmark

`reports/candidate_benchmark.json` keeps the four-family single-split benchmark. It is development evidence, not the sole deployment rule.

## Expanding temporal backtest

```bash
python -m scripts.run_temporal_backtest \
  --data-path data/processed/flights_processed.parquet \
  --max-rows 200000 \
  --n-splits 4 \
  --candidate-profile linear
```

Each fold contains an earlier training window, a later validation/calibration block and a subsequent unseen test block. All cohort features are fit inside the fold.

The committed report is in `reports/temporal_backtest.json` and `.md`.

## Calibration report

`reports/calibration_report.json` compares the raw selected-model score with the post-calibrated probability on the held-out test block.

## Confidence intervals

The training command supports optional bootstrap intervals:

```bash
python -m scripts.train_model --bootstrap-samples 200
```

They are intentionally optional because repeated scoring can increase runtime substantially.

## Interpreting the result

FlightRisk does not optimize for accuracy. A naive majority classifier could achieve high accuracy while failing to identify delayed flights. The central questions are:

- Are delayed flights concentrated near the top of the queue?
- Are displayed probabilities aligned with observed frequencies?
- Does the result survive later temporal blocks?
