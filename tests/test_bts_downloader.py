from pathlib import Path

from scripts.download_bts_data import build_bts_prezip_url, manual_download_instructions, parse_months


def test_build_bts_prezip_url_contains_expected_file_name():
    url = build_bts_prezip_url(2024, 1)
    assert url.startswith("https://transtats.bts.gov/PREZIP/")
    assert "2024_1.zip" in url
    assert "On_Time_Reporting_Carrier_On_Time_Performance" in url


def test_parse_months_deduplicates_and_sorts():
    assert parse_months([3, 1, 2, 2]) == [1, 2, 3]


def test_manual_download_instructions_are_actionable():
    text = manual_download_instructions(2024, 1, Path("data/raw"))
    assert "BTS TranStats download page" in text
    assert "2024" in text
    assert "FlightDate" in text
    assert "python -m scripts.run_real_data_demo" in text
