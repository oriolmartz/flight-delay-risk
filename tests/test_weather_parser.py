import pandas as pd

from src.weather.parser import normalize_weather_frame


def test_normalizes_common_noaa_fields() -> None:
    raw = pd.DataFrame(
        {
            "STATION": ["74486094789"],
            "DATE": ["2024-01-01T12:00:00"],
            "TMP": ["+0123,1"],
            "DEW": ["+0050,1"],
            "WND": ["180,1,N,0050,1"],
            "VIS": ["016000,1,N,1"],
            "SLP": ["10123,1"],
        }
    )
    result = normalize_weather_frame(raw)
    assert result.loc[0, "temperature_c"] == 12.3
    assert result.loc[0, "dew_point_c"] == 5.0
    assert result.loc[0, "wind_speed_mps"] == 5.0
    assert result.loc[0, "visibility_m"] == 16000
    assert result.loc[0, "sea_level_pressure_hpa"] == 1012.3
    assert str(result.loc[0, "observation_time_utc"].tz) == "UTC"
