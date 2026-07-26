"""Download reproducible NOAA Global Hourly station-year extracts."""
from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"


def station_year_url(station_id: str, year: int) -> str:
    if year < 1900 or year > 2100:
        raise ValueError("year is outside the supported range")
    station = str(station_id).strip()
    if not station.isdigit():
        raise ValueError("station_id must contain digits only")
    return BASE_URL.format(year=year, station_id=station)


def download_station_year(
    station_id: str,
    year: int,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{station_id}_{year}.csv"
    if target.exists() and not overwrite:
        return target
    request = Request(station_year_url(station_id, year), headers={"User-Agent": "flight-delay-risk/1.5"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"NOAA download failed with HTTP {exc.code}: {request.full_url}") from exc
    except URLError as exc:
        raise RuntimeError(f"NOAA download failed: {exc.reason}") from exc
    temporary = target.with_suffix(".csv.part")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target
