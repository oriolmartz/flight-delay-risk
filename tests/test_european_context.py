import pytest

from src.reference.european_context import lookup_european_context, summarize_european_context


@pytest.fixture(autouse=True)
def allow_sample(monkeypatch):
    monkeypatch.setenv("FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT", "1")


def test_lookup_european_context_exact_match():
    ctx = lookup_european_context("VY", "BCN", "AMS", 7)
    assert ctx.status == "matched"
    assert ctx.matched_level == "airline_route_month"
    assert ctx.pct_flights_15min_late is not None


def test_lookup_european_context_missing_route():
    ctx = lookup_european_context("IB", "BCN", "ZRH", 2)
    assert ctx.status in {"missing", "unavailable"}


def test_summarize_european_context():
    summary = summarize_european_context()
    assert summary["available"] is True
    assert summary["rows"] >= 1
    assert "BCN-AMS" in summary["routes"]
