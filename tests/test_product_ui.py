from pathlib import Path


def test_dashboard_is_native_streamlit_product_surface():
    code = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "streamlit.components" not in code
    assert "components.html" not in code
    assert "st.sidebar" not in code
    assert 'initial_sidebar_state="collapsed"' in code
    assert "st.tabs" in code
    assert "date_input" in code
    assert "time_input" in code
    assert "rank_dataframe" in code


def test_dashboard_exposes_four_decision_surfaces():
    copy = Path("app/dashboard/i18n.py").read_text(encoding="utf-8")
    assert "Analyze flight" in copy
    assert "Rank schedule" in copy
    assert "Validation" in copy
    assert "Model & operations" in copy
    assert "attention before departure" in copy


def test_dashboard_uses_personal_visual_system():
    theme = Path("app/dashboard/theme.py").read_text(encoding="utf-8")
    dashboard = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "--fr-bg: #f4f9ff" in theme
    assert "--fr-navy: #164a73" in theme
    assert "--fr-amber: #b7791f" in theme
    assert "Oriol Martínez" in dashboard or "Oriol Martínez" in Path("app/dashboard/i18n.py").read_text(encoding="utf-8")
    assert "fr-flight-card" in dashboard
    assert "FLIGHT DELAY RISK" in dashboard
    assert "PR-AUC / prevalence" in Path("app/dashboard/i18n.py").read_text(encoding="utf-8")


def test_readme_is_product_and_recruiter_friendly():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Pre-departure decision support for limited airline operations capacity" in readme
    assert "Product workflow" in readme
    assert "Validation design" in readme
    assert "Operational result" in readme
    assert "Built by **Oriol Martínez**" in readme


def test_dashboard_explains_metrics_without_readme_lookup():
    copy = Path("app/dashboard/i18n.py").read_text(encoding="utf-8")
    dashboard = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "How to read the metrics" in copy
    assert "Cómo leer las métricas" in copy
    assert "Higher is better" in copy
    assert "Cuanto menor, mejor" in copy
    assert "Advanced validation diagnostics" in copy
    assert "Detalles avanzados del modelo" in copy
    assert "_metric_cards" in dashboard
    assert "technical_explanation" in dashboard


def test_primary_flight_summary_hides_raw_model_jargon():
    dashboard = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    prediction_block = dashboard.split("def _render_prediction", 1)[1].split("# Schedule upload", 1)[0]
    assert "_metric_cards" in prediction_block
    assert "advanced_details" in prediction_block
    assert "raw_score" in prediction_block
    assert prediction_block.index("advanced_details") < prediction_block.rindex("raw_score")


def test_dashboard_exposes_weather_evidence_and_network_layers():
    dashboard = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    copy = Path("app/dashboard/i18n.py").read_text(encoding="utf-8")
    assert "_render_weather_context" in dashboard
    assert "weather_enhanced_prediction" in dashboard
    assert "Airport × hour matrix" in copy
    assert "Weather-associated uplift" in copy
    assert "historical replay evidence" in copy
    assert "Future flight · schedule-only prediction" in copy
    assert "live, versioned forecast" in copy
    assert "cleaned flight observations" in copy
    assert "PROMINENT_AIRPORT_LABELS" in dashboard
    assert "one row per airport" in copy.lower()
    assert "SEPARATE FROM THE RELEASE SCORE" in copy
    assert "One official prediction, one replay diagnostic" in copy
    assert "Do not add this delta to the official prediction" in copy
    weather_block = dashboard.split("def _render_weather_context", 1)[1].split("def _render_prediction", 1)[0]
    assert "fr-weather-signal-card" in weather_block
    assert "paired_base_probability" not in weather_block
    assert "weather_probability" not in weather_block
    assert "comparison_formula" not in weather_block
    assert "_metric_cards" not in weather_block
    assert "delta * 100" in weather_block
    assert 'mode != "historical_replay"' in weather_block
    assert "future_schedule_only" in weather_block
