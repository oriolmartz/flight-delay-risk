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
    assert "--fr-bg: #f3efe6" in theme
    assert "--fr-navy: #17304f" in theme
    assert "--fr-amber: #bb7a24" in theme
    assert "Built by Oriol Martínez" in dashboard
    assert "fr-flight-card" in dashboard


def test_readme_is_product_and_recruiter_friendly():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Pre-departure flight-delay risk workbench" in readme
    assert "Product tour" in readme
    assert "Leakage contract" in readme
    assert "Honest result" in readme
    assert "Built by **Oriol Martínez**" in readme
