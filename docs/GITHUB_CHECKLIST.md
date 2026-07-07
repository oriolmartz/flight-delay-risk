# GitHub release checklist

Before pushing FlightRisk, run:

```bash
python -m compileall -q app src scripts tests
pytest -q
python -m scripts.run_local_demo
```

If you trained on real BTS data, verify:

```bash
python -m scripts.run_real_data_demo --max-rows 200000
python -m uvicorn app.api.main:app --reload --port 8000
python -m streamlit run app/dashboard/streamlit_app.py
```

Do not commit:

- `data/raw/*.csv`
- `data/processed/*`
- `models/*.joblib`
- `.venv/`
- `.env`
- `monitoring/prediction_log.csv`

Good assets to commit:

- `reports/metrics.json`
- `reports/error_analysis.md`
- `reports/feature_importance.csv`
- screenshots/GIFs under `docs/assets/`
- source code, tests and documentation

Suggested repo name:

```text
flightrisk-ml
```
