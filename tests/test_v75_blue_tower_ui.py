from pathlib import Path


def test_dashboard_is_native_blue_tower_not_embedded_landing():
    code = Path("app/dashboard/streamlit_app.py").read_text()
    assert "streamlit.components" not in code
    assert "components.html" not in code
    assert "st.sidebar" not in code
    assert 'initial_sidebar_state="collapsed"' in code
    assert "fr-tower" in code
    assert "language_selector" in code


def test_dashboard_states_prediction_target_training_and_europe_truth():
    code = Path("app/dashboard/streamlit_app.py").read_text()
    assert "P(ArrDel15 = 1)" in code
    assert "probability of arriving 15+ minutes late" in code or "probability that a scheduled flight arrives 15+ minutes late" in code
    assert "BTS U.S. On-Time Performance flight-level data" in code
    assert "UK CAA" in code
    assert "not the core flight-level training set" in code
