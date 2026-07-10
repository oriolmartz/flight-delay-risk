# Interview notes

## How to describe FlightRisk

FlightRisk is an end-to-end ML engineering system for ranking flight delay risk using real aviation datasets.

- U.S. path: real BTS flight-level data and `ArrDel15` prediction.
- Europe path: real UK CAA aggregate punctuality modeling.
- Serving: FastAPI API and Streamlit cockpit.
- MLOps: prediction logging, PSI drift reference, Docker/CI and AWS/ECS skeleton.

## Expected technical questions

### Why only one month in the baseline run?

The pipeline supports many monthly BTS files, but one month is a controlled baseline for fast reproducibility. For stronger evaluation, I would use 6-24 months and run rolling temporal backtests.

### Why not random k-fold?

Flight data is temporal. Random k-fold would leak future distribution into past folds. I use time-aware splits and the repository includes expanding-window backtests.

### Why are the U.S. metrics modest?

Delay prediction with schedule-only features is hard. Weather, aircraft rotation, ATC/NAS conditions and late aircraft propagation are not included. The system is more about ML engineering and risk ranking than claiming very high predictive accuracy.

### Why use Lift@10?

For risk-ranking products, the question is whether the model concentrates bad outcomes at the top of the risk list. Lift@10 tells us whether the highest-risk decile contains delayed flights at a higher rate than random selection.

### Is Europe the same model as U.S.?

No. The U.S. path is flight-level delay prediction. The European path is aggregate punctuality modeling from UK CAA data. They are intentionally separated in the UI and docs.

### Is it deployed?

The repo includes Docker, CI and AWS/ECS skeletons. A live public endpoint is the next production step.
