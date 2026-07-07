from pathlib import Path


def test_dashboard_is_native_simple_delay_probability_ui():
    code = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert "streamlit.components" not in code
    assert "components.html" not in code
    assert "st.sidebar" not in code
    assert 'initial_sidebar_state="collapsed"' in code
    assert "fr-tower" in code
    assert "language_selector" in code
    assert "rank_dataframe" in code


def test_dashboard_defaults_to_english_and_states_core_truth():
    code = Path("app/dashboard/streamlit_app.py").read_text(encoding="utf-8")
    assert '["English", "Español"], index=0' in code
    assert "P(ArrDel15 = 1)" in code
    assert "probability that a scheduled flight arrives 15+ minutes late" in code
    assert "BTS U.S. On-Time Performance flight-level data" in code
    assert "UK CAA aggregate punctuality data" in code
    assert "not the core flight-level training set" in code


def test_readme_has_simple_project_framing():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "probability that a scheduled flight arrives 15+ minutes late" in readme
    assert "English is the default UI language" in readme
    assert "U.S. BTS On-Time Performance flight-level data" in readme
