"""Download or prepare BTS monthly on-time performance data.

Important BTS note
------------------
The official TranStats UI exposes monthly CSV downloads, but the generated
``/PREZIP/`` filenames are session/generated artifacts rather than a stable
public API. This script still attempts the known direct PREZIP pattern because
it sometimes works for mirrored exports, but it now fails gracefully with exact
manual download instructions instead of throwing an opaque HTTP traceback.

Recommended production-safe workflow:
    1. Download monthly CSVs from the official TranStats page.
    2. Place them in ``data/raw/``.
    3. Run ``python -m scripts.run_real_data_demo`` without ``--download``.

Example manual-data run:
    python -m scripts.run_real_data_demo

Example best-effort direct attempt:
    python -m scripts.download_bts_data --year 2024 --months 1 2 3 --output-dir data/raw
"""
from __future__ import annotations

import argparse
import io
import textwrap
import zipfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from src.data.load_data import normalize_columns
from src.utils.logging import get_logger

logger = get_logger(__name__)

BTS_DOWNLOAD_PAGE = "https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ"
BTS_PREZIP_BASE = "https://transtats.bts.gov/PREZIP"
BTS_PREZIP_NAME = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip"

KEEP_COLUMNS = [
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
]

RECOMMENDED_BTS_FIELDS = [
    "FlightDate",
    "Year",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
]


class BTSDownloadError(RuntimeError):
    """Raised when BTS data cannot be downloaded automatically."""


def build_bts_prezip_url(year: int, month: int) -> str:
    """Build the best-effort BTS TranStats PREZIP URL.

    TranStats can generate files with non-stable names. Treat this as a
    convenience attempt, not a guaranteed API contract.
    """
    filename = BTS_PREZIP_NAME.format(year=year, month=month)
    return f"{BTS_PREZIP_BASE}/{quote(filename)}"


def manual_download_instructions(year: int, month: int, output_dir: Path) -> str:
    fields = "\n".join(f"  - {field}" for field in RECOMMENDED_BTS_FIELDS)
    return textwrap.dedent(
        f"""
        Could not download BTS {year}-{month:02d} automatically.

        TranStats monthly CSV downloads are generated through the official web
        download page and their PREZIP URLs are not stable. Do this once:

        1. Open the official BTS TranStats download page:
           {BTS_DOWNLOAD_PAGE}

        2. Select:
           - Filter Year: {year}
           - Filter Period: {month}
           - Table: Reporting Carrier On-Time Performance (1987-present)

        3. Select at least these fields:
{fields}

        4. Click Download / Prezipped File.

        5. Extract the CSV if needed and place it here:
           {Path(output_dir).resolve()}

        6. Re-run without --download:
           python -m scripts.run_real_data_demo
        """
    ).strip()


def _read_first_csv_from_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Downloaded BTS zip did not contain a CSV file.")
        csv_name = csv_names[0]
        logger.info("Reading %s from BTS zip", csv_name)
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh, low_memory=False)


def _normalize_and_write_bts_frame(
    df: pd.DataFrame,
    year: int,
    month: int,
    output_dir: Path,
    max_rows: int | None = None,
) -> Path:
    df = normalize_columns(df)

    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("BTS file is missing expected columns after normalization: %s", missing)
    df = df[keep]

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).sort_index().reset_index(drop=True)
        logger.info("Sampled %d rows for a lightweight portfolio run", max_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"bts_on_time_{year}_{month:02d}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path


def download_month(year: int, month: int, output_dir: Path, max_rows: int | None = None) -> Path:
    """Best-effort download of one BTS month and write a compact normalized CSV.

    If TranStats returns 404/403 or an invalid zip, a BTSDownloadError with
    manual instructions is raised. This avoids hiding the real issue behind a
    raw requests traceback.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = build_bts_prezip_url(year, month)
    logger.info("Attempting BTS %04d-%02d direct PREZIP download from %s", year, month, url)

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        df = _read_first_csv_from_zip(response.content)
    except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
        raise BTSDownloadError(manual_download_instructions(year, month, output_dir)) from exc

    return _normalize_and_write_bts_frame(df, year, month, output_dir, max_rows)


def parse_months(values: list[int]) -> list[int]:
    months = sorted(set(values))
    invalid = [m for m in months if m < 1 or m > 12]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid month values: {invalid}")
    return months


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official BTS monthly CSV data when direct PREZIP is available.")
    parser.add_argument("--year", type=int, required=True, help="BTS data year, e.g. 2024.")
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        required=True,
        help="One or more month numbers, e.g. --months 1 2 3.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--max-rows-per-month",
        type=int,
        default=None,
        help="Optional row sample per month for lightweight local runs.",
    )
    args = parser.parse_args()

    months = parse_months(args.months)
    written = []
    try:
        for month in months:
            written.append(download_month(args.year, month, args.output_dir, args.max_rows_per_month))
    except BTSDownloadError as exc:
        print(str(exc))
        raise SystemExit(2) from exc

    logger.info("Downloaded %d month(s): %s", len(written), [str(p) for p in written])


if __name__ == "__main__":
    main()
