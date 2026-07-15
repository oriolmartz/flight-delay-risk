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
    assert "Pre-departure flight-delay risk workbench" in readme
    assert "Product tour" in readme
    assert "Leakage contract" in readme
    assert "Honest result" in readme
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
