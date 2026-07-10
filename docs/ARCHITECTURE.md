# FlightRisk architecture

FlightRisk v3 is organized as a production-shaped ML system rather than a notebook-only experiment.

![FlightRisk architecture](assets/architecture.svg)

## High-level flow

```text
Official BTS monthly CSVs
  → schema normalization
  → cleaning + validation
  → leakage column removal
  → time-aware train / validation / test split
  → pre-flight feature engineering
  → historical aggregate features fit on model-training data only
  → candidate model training
  → validation-based model selection
  → validation-based decision-threshold tuning
  → held-out test evaluation reports
  → model artifact with metadata + threshold
  → FastAPI and Streamlit serving
```

## Layers

### Data layer

Files:

```text
scripts/download_bts_data.py
scripts/prepare_data.py
src/data/load_data.py
src/data/clean.py
src/data/io.py
src/data/split.py
```

Responsibilities:

- Download/accept real BTS monthly CSVs
- Normalize inconsistent BTS column names
- Validate required fields
- Filter rows with missing target
- Filter cancelled/diverted rows
- Remove leakage columns
- Write processed data as Parquet or CSV fallback
- Split chronologically when `FlightDate` exists

### Feature layer

Files:

```text
src/features/build_features.py
src/features/historical_aggregates.py
```

Responsibilities:

- Convert scheduled HHMM times to hours
- Build route identifiers
- Add weekend flag
- Fit historical carrier/route/origin/destination delay rates on model-training rows only
- Apply aggregate fallbacks to unseen inference values

### Model layer

Files:

```text
src/models/train.py
src/models/evaluate.py
src/models/thresholding.py
src/models/registry.py
src/models/predict.py
```

Responsibilities:

- Build preprocessing pipelines
- Train Logistic Regression baseline
- Train tree-based candidates: Random Forest and Extra Trees
- Select the deployed candidate using validation PR-AUC by default
- Tune the decision threshold on validation predictions for F1
- Evaluate ROC-AUC, PR-AUC, F1, precision, recall and calibration on the held-out test split
- Save model artifact with pipeline + aggregates + metrics + metadata + decision threshold
- Run single/batch predictions without train/serve skew

### Serving layer

Files:

```text
app/api/main.py
app/dashboard/streamlit_app.py
app/services/prediction_service.py
app/schemas.py
```

Responsibilities:

- Load model artifact once
- Expose `/health`, `/model/info`, `/model/card`, `/predict`, `/predict/batch`
- Provide a blue recruiter-friendly dashboard
- Display model metrics, selected model, tuned threshold, prediction probability, risk level and risk drivers

## Key design choice: leakage-safe historical aggregates

Historical route/carrier/airport delay rates can be powerful features, but they are unsafe if computed from all data before splitting. FlightRisk avoids this by:

1. splitting data first,
2. fitting aggregate lookup tables on model-training rows only,
3. applying those lookups to validation/test/inference rows,
4. using a global fallback for unseen keys.

This mirrors how risk features are usually built in real tabular ML systems.

## Key design choice: validation selection before test reporting

FlightRisk v3 separates model selection from final test reporting:

1. train candidate models on the model-training split,
2. compare candidates on the validation split,
3. tune the decision threshold on validation predictions,
4. report final metrics once on the held-out test split.

This avoids tuning directly on the test set and makes the reported metrics more credible.
