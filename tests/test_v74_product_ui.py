from pathlib import Path


def test_dashboard_is_streamlit_native_not_embedded_html_landing():
    code = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "streamlit.components" not in code
    assert "components.html" not in code
    assert "st.sidebar" not in code
    assert 'initial_sidebar_state="collapsed"' in code
    assert "rank_dataframe" in code


def test_dashboard_has_simple_delay_probability_message():
    code = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "Estimate delay probability" in code or "Estimate the probability" in code
    assert "Batch mode" in code
    assert "Model details" in code
