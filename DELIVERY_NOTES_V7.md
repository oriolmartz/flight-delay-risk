# FlightRisk v7.9.2 delivery notes

This release fixes the two visual regressions found in v7.9.1: the technical section no longer uses black code-block panels for architecture/pipeline content, and the post-prediction chart is now a readable light horizontal probability split.

Run:

```powershell
streamlit run app/dashboard/streamlit_app.py
```

Sample training:

```powershell
python -m scripts.run_real_data_demo --selection-metric pr_auc --bootstrap-samples 0 --max-rows-per-month 5000
```
