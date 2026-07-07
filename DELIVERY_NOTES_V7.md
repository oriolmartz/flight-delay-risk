# Delivery notes

This version intentionally simplifies the FlightRisk narrative.

The first screen now answers:

1. What does it predict?
   - P(ArrDel15 = 1): probability of arriving 15+ minutes late.

2. What data trained the main model?
   - U.S. BTS On-Time Performance flight-level data.

3. What does batch mode do?
   - Applies the same probability model to many flights and sorts by risk.

4. What is Europe?
   - Experimental UK CAA aggregate punctuality context only.

English is the default language.


## v7.6 sampling fix

`--max-rows-per-month` now applies when reading existing CSV files from `data/raw/`, not only during BTS auto-download. This means quick experiments no longer load all 7M+ rows before sampling.

Fast smoke training:

```powershell
python -m scripts.run_real_data_demo --selection-metric pr_auc --bootstrap-samples 0 --max-rows-per-month 5000
```

Expected first log with 12 monthly files: around `Combined 12 files into 60000 total rows with max_rows_per_file=5000`.
