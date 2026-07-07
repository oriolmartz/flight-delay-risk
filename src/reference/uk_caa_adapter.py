"""UK CAA punctuality adapter for FlightRisk.

FlightRisk v5.3 uses this adapter to turn UK Civil Aviation Authority
punctuality CSV exports into the canonical European context schema used by
`src.reference.european_context`.

Why an adapter instead of hard-coded real data?
- UK CAA publishes punctuality files as monthly CSV/XLSX resources.
- Public file layouts may vary slightly by year/month/report type.
- The repo should not ship bulky official datasets or imply redistribution
  rights; instead it ships a robust parser and a documented drop-in workflow.

Canonical output columns:
    year, month, airline, origin, destination, airport_pair,
    avg_arrival_delay_min, pct_flights_15min_late, cancelled_pct, source
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


CANONICAL_COLUMNS = [
    "year",
    "month",
    "airline",
    "origin",
    "destination",
    "airport_pair",
    "avg_arrival_delay_min",
    "pct_flights_15min_late",
    "cancelled_pct",
    "source",
    "number_flights_matched",
]

UK_AIRPORT_CITY_TO_IATA = {
    "LONDON HEATHROW": "LHR",
    "HEATHROW": "LHR",
    "LONDON GATWICK": "LGW",
    "GATWICK": "LGW",
    "MANCHESTER": "MAN",
    "STANSTED": "STN",
    "LONDON STANSTED": "STN",
    "LUTON": "LTN",
    "LONDON LUTON": "LTN",
    "EDINBURGH": "EDI",
    "BIRMINGHAM": "BHX",
    "GLASGOW": "GLA",
    "BRISTOL": "BRS",
    "NEWCASTLE": "NCL",
    "LIVERPOOL": "LPL",
}

COMMON_AIRLINE_NAME_TO_CODE = {
    "BRITISH AIRWAYS": "BA",
    "EASYJET": "U2",
    "RYANAIR": "FR",
    "VIRGIN ATLANTIC": "VS",
    "JET2": "LS",
    "TUI AIRWAYS": "BY",
    "KLM": "KL",
    "AIR FRANCE": "AF",
    "LUFTHANSA": "LH",
    "IBERIA": "IB",
    "VUELING": "VY",
    "AER LINGUS": "EI",
    "TAP AIR PORTUGAL": "TP",
    "SWISS": "LX",
    "EMIRATES": "EK",
    "QATAR AIRWAYS": "QR",
    "TURKISH AIRLINES": "TK",
}

EUROPE_DESTINATION_HINTS = {
    "AMSTERDAM": "AMS",
    "BARCELONA": "BCN",
    "MADRID": "MAD",
    "PARIS": "CDG",
    "PARIS CDG": "CDG",
    "CHARLES DE GAULLE": "CDG",
    "FRANKFURT": "FRA",
    "MUNICH": "MUC",
    "ROME": "FCO",
    "ROME FIUMICINO": "FCO",
    "MILAN": "MXP",
    "LISBON": "LIS",
    "DUBLIN": "DUB",
    "COPENHAGEN": "CPH",
    "STOCKHOLM": "ARN",
    "ZURICH": "ZRH",
    "VIENNA": "VIE",
    "ATHENS": "ATH",
}

COLUMN_SYNONYMS = {
    "airline": ["airline", "carrier", "operator", "operating_airline", "airline_name"],
    "origin": ["origin", "from", "departure_airport", "airport", "uk_airport", "reporting_airport"],
    "destination": ["destination", "dest", "to", "arrival_airport", "foreign_airport", "route_destination", "origin_destination"],
    "route": ["route", "airport_pair", "city_pair"],
    "avg_arrival_delay_min": [
        "avg_arrival_delay_min",
        "average_arrival_delay",
        "average_arrival_delay_mins",
        "avg_delay",
        "average_delay_mins",
        "mean_arrival_delay",
    ],
    "pct_flights_15min_late": [
        "pct_flights_15min_late",
        "percent_15min_late",
        "percent_flights_15_minutes_late",
        "arrivals_15_minutes_late_pct",
        "early_to_15_mins_late_percent",
        "percentage_late",
        "late_15_pct",
        "flights_more_than_15_minutes_late_percent",
        "more_than_15_mins_late_percent",
    ],
    "cancelled_pct": ["cancelled_pct", "cancellation_pct", "cancelled_percent", "percent_cancelled"],
    "year": ["year"],
    "month": ["month", "reporting_month"],
    "reporting_period": ["reporting_period"],
    "arrival_departure": ["arrival_departure"],
    "scheduled_charter": ["scheduled_charter"],
    "number_flights_matched": ["number_flights_matched", "matched_flights", "flights_matched", "total_flights"],
}


@dataclass
class AdapterReport:
    input_files: int
    output_rows: int
    output_path: str
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "input_files": self.input_files,
            "output_rows": self.output_rows,
            "output_path": self.output_path,
            "warnings": self.warnings,
        }


def _clean_column_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[%()+/\-]+", " ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _find_column(df: pd.DataFrame, logical_name: str) -> str | None:
    synonyms = COLUMN_SYNONYMS.get(logical_name, [logical_name])
    columns = list(df.columns)
    for candidate in synonyms:
        c = _clean_column_name(candidate)
        if c in columns:
            return c

    # fuzzy fallback: every token in candidate appears in column
    for candidate in synonyms:
        tokens = [t for t in _clean_column_name(candidate).split("_") if t]
        for col in columns:
            if all(t in col for t in tokens):
                return col
    return None


def _extract_year_month_from_filename(path: Path) -> tuple[int | None, int | None]:
    match = re.search(r"(20\d{2})[_\- ]?([01]\d)", path.stem)
    if not match:
        return None, None
    year = int(match.group(1))
    month = int(match.group(2))
    if 1 <= month <= 12:
        return year, month
    return year, None


def _to_airline_code(value: object) -> str:
    raw = str(value).strip().upper()
    if not raw or raw == "NAN":
        return "UNKNOWN"
    if re.fullmatch(r"[A-Z0-9]{2,3}", raw):
        return raw
    return COMMON_AIRLINE_NAME_TO_CODE.get(raw, raw[:3].replace(" ", ""))


def _to_airport_code(value: object, *, default: str = "UNKNOWN") -> str:
    raw = str(value).strip().upper()
    if not raw or raw == "NAN":
        return default
    if re.fullmatch(r"[A-Z]{3}", raw):
        return raw
    if raw in UK_AIRPORT_CITY_TO_IATA:
        return UK_AIRPORT_CITY_TO_IATA[raw]
    if raw in EUROPE_DESTINATION_HINTS:
        return EUROPE_DESTINATION_HINTS[raw]
    # CAA fields sometimes contain "London Heathrow - Amsterdam" style strings.
    for name, code in {**UK_AIRPORT_CITY_TO_IATA, **EUROPE_DESTINATION_HINTS}.items():
        if name in raw:
            return code
    return raw[:3].replace(" ", "")


def _split_route(value: object) -> tuple[str, str] | None:
    raw = str(value).strip().upper()
    if not raw or raw == "NAN":
        return None
    for sep in ["-", "–", "—", " TO ", " / "]:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) >= 2:
                return _to_airport_code(parts[0]), _to_airport_code(parts[1])
    return None


def normalize_uk_caa_punctuality_file(path: str | Path, *, default_year: int | None = None, default_month: int | None = None) -> pd.DataFrame:
    """Normalize one UK CAA punctuality CSV file into FlightRisk's European context schema.

    Handles both the official CAA-style monthly columns documented in public
    examples (`reporting_period`, `reporting_airport`, `origin_destination`,
    `arrival_departure`, `scheduled_charter`, `number_flights_matched`,
    `average_delay_mins`) and cleaner pre-normalized variants.
    """
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={c: _clean_column_name(c) for c in df.columns})

    col_arr_dep = _find_column(df, "arrival_departure")
    if col_arr_dep:
        # Keep scheduled departures where available. This makes route direction explicit.
        df = df[df[col_arr_dep].astype(str).str.upper().str.strip().eq("D")]

    col_sched = _find_column(df, "scheduled_charter")
    if col_sched:
        df = df[df[col_sched].astype(str).str.upper().str.strip().eq("S")]

    year_from_name, month_from_name = _extract_year_month_from_filename(path)
    year = default_year or year_from_name
    month = default_month or month_from_name

    col_period = _find_column(df, "reporting_period")
    col_year = _find_column(df, "year")
    col_month = _find_column(df, "month")
    col_airline = _find_column(df, "airline")
    col_origin = _find_column(df, "origin")
    col_destination = _find_column(df, "destination")
    col_route = _find_column(df, "route")
    col_avg_delay = _find_column(df, "avg_arrival_delay_min")
    col_late_pct = _find_column(df, "pct_flights_15min_late")
    col_cancelled = _find_column(df, "cancelled_pct")
    col_n = _find_column(df, "number_flights_matched")

    output_rows: list[dict] = []
    for _, row in df.iterrows():
        current_year = year
        current_month = month
        if col_period and pd.notna(row[col_period]):
            period = str(row[col_period]).strip()
            match = re.search(r"(20\d{2})\D?([01]\d)", period)
            if match:
                current_year = int(match.group(1))
                current_month = int(match.group(2))
        if col_year and pd.notna(row[col_year]):
            current_year = int(row[col_year])
        if col_month and pd.notna(row[col_month]):
            current_month = int(row[col_month])
        if current_year is None or current_month is None:
            continue

        airline = _to_airline_code(row[col_airline]) if col_airline else "UNKNOWN"
        origin = _to_airport_code(row[col_origin]) if col_origin else "UNKNOWN"
        destination = _to_airport_code(row[col_destination]) if col_destination else "UNKNOWN"
        if col_route and (origin == "UNKNOWN" or destination == "UNKNOWN"):
            route = _split_route(row[col_route])
            if route:
                origin, destination = route

        avg_delay = pd.to_numeric(row[col_avg_delay], errors="coerce") if col_avg_delay else pd.NA
        late_pct = pd.to_numeric(row[col_late_pct], errors="coerce") if col_late_pct else pd.NA
        cancelled_pct = pd.to_numeric(row[col_cancelled], errors="coerce") if col_cancelled else pd.NA
        matched = pd.to_numeric(row[col_n], errors="coerce") if col_n else pd.NA

        # CAA older files often provide early-to-15-mins-late %, not 15+ late %.
        # If the column name is early_to_15..., convert to late share = 1 - on_time_share.
        if col_late_pct and "early_to_15" in col_late_pct and pd.notna(late_pct):
            late_pct = 100 - late_pct if late_pct > 1 else 1 - late_pct

        if pd.notna(late_pct) and late_pct > 1:
            late_pct = late_pct / 100
        if pd.notna(cancelled_pct) and cancelled_pct > 1:
            cancelled_pct = cancelled_pct / 100

        if origin == "UNKNOWN" or destination == "UNKNOWN":
            continue

        output_rows.append({
            "year": int(current_year),
            "month": int(current_month),
            "airline": airline,
            "origin": origin,
            "destination": destination,
            "airport_pair": f"{origin}-{destination}",
            "avg_arrival_delay_min": None if pd.isna(avg_delay) else float(avg_delay),
            "pct_flights_15min_late": None if pd.isna(late_pct) else float(late_pct),
            "cancelled_pct": None if pd.isna(cancelled_pct) else float(cancelled_pct),
            "source": f"UK_CAA::{path.name}",
            "number_flights_matched": None if pd.isna(matched) else int(matched),
        })

    return pd.DataFrame(output_rows, columns=CANONICAL_COLUMNS)

def build_uk_caa_context_dataset(
    raw_dir: str | Path = "data/europe/uk_caa_raw",
    output_path: str | Path = "data/europe/europe_punctuality_context.csv",
) -> AdapterReport:
    """Normalize all UK CAA CSV files in a directory into one context dataset."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    frames: list[pd.DataFrame] = []
    files = sorted(raw_dir.glob("*.csv"))

    for file in files:
        try:
            frame = normalize_uk_caa_punctuality_file(file)
            if frame.empty:
                warnings.append(f"{file.name}: no usable rows after normalization")
            else:
                frames.append(frame)
        except Exception as exc:
            warnings.append(f"{file.name}: {exc}")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates()
    else:
        out = pd.DataFrame(columns=CANONICAL_COLUMNS)

    out.to_csv(output_path, index=False)

    return AdapterReport(
        input_files=len(files),
        output_rows=int(len(out)),
        output_path=str(output_path),
        warnings=warnings,
    )
