## v7.7 lighter portfolio UI

- Reworked the dashboard from dark AI/HUD styling to a lighter aviation portfolio look.
- Added a clean airplane visual to the hero section.
- Reduced the main headline size and improved spacing/readability.
- Kept the simple delay-probability framing: P(ArrDel15 = 1).
- Preserved Streamlit-native structure, no sidebar and no embedded landing-page component.

## Simple Delay Probability UI

- Rebuilt the dashboard around one clear concept: probability of 15+ minute arrival delay.
- Set English as the default UI language.
- Kept a visible EN / ES language selector.
- Moved single-flight prediction above batch ranking.
- Reframed batch mode as a secondary sorting workflow.
- Reframed Europe as experimental UK CAA aggregate context, not the core trained model.
- Kept the blue aviation aesthetic and control-tower background.
- Removed sidebar usage and avoided embedded landing-page HTML.


## v7.6 sampling fix

`--max-rows-per-month` now applies when reading existing CSV files from `data/raw/`, not only during BTS auto-download. This means quick experiments no longer load all 7M+ rows before sampling.

Fast smoke training:

```powershell
python -m scripts.run_real_data_demo --selection-metric pr_auc --bootstrap-samples 0 --max-rows-per-month 5000
```

Expected first log with 12 monthly files: around `Combined 12 files into 60000 total rows with max_rows_per_file=5000`.
