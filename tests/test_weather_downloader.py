import pytest

from src.weather.downloader import station_year_url


def test_station_year_url_is_reproducible() -> None:
    assert station_year_url("74486094789", 2024).endswith("/2024/74486094789.csv")


def test_station_year_url_rejects_invalid_station() -> None:
    with pytest.raises(ValueError, match="digits only"):
        station_year_url("KJFK", 2024)
