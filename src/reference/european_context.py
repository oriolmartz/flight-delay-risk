"""Real European aggregated punctuality context layer for FlightRisk v6.

The production European mode is real-data-only: it consumes a generated context
CSV built from official/public European punctuality sources (currently UK CAA).
The bundled sample CSV exists only for tests and local UI smoke tests when the
`FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT=1` environment variable is set.

Canonical CSV schema:
    year, month, airline, origin, destination, airport_pair,
    avg_arrival_delay_min, pct_flights_15min_late, cancelled_pct, source,
    number_flights_matched
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd


DEFAULT_EUROPE_CONTEXT_PATH = Path("data/europe/europe_punctuality_context.csv")
SAMPLE_EUROPE_CONTEXT_PATH = Path("data/europe/europe_punctuality_sample.csv")


def sample_context_allowed() -> bool:
    return os.getenv("FLIGHTRISK_ALLOW_SAMPLE_EUROPE_CONTEXT", "0").lower() in {"1", "true", "yes"}


def resolve_context_path(path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH) -> Path:
    path = Path(path)
    if path.exists():
        return path
    if sample_context_allowed() and SAMPLE_EUROPE_CONTEXT_PATH.exists():
        return SAMPLE_EUROPE_CONTEXT_PATH
    return path


@dataclass
class EuropeanPunctualityContext:
    status: str
    source: str
    year: int | None = None
    month: int | None = None
    airline: str | None = None
    origin: str | None = None
    destination: str | None = None
    airport_pair: str | None = None
    avg_arrival_delay_min: float | None = None
    pct_flights_15min_late: float | None = None
    cancelled_pct: float | None = None
    number_flights_matched: int | None = None
    matched_level: str = "none"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "source": self.source,
            "year": self.year,
            "month": self.month,
            "airline": self.airline,
            "origin": self.origin,
            "destination": self.destination,
            "airport_pair": self.airport_pair,
            "avg_arrival_delay_min": self.avg_arrival_delay_min,
            "pct_flights_15min_late": self.pct_flights_15min_late,
            "cancelled_pct": self.cancelled_pct,
            "number_flights_matched": self.number_flights_matched,
            "matched_level": self.matched_level,
        }


def load_european_context(path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH) -> pd.DataFrame:
    path = resolve_context_path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "year", "month", "airline", "origin", "destination", "airport_pair",
                "avg_arrival_delay_min", "pct_flights_15min_late", "cancelled_pct",
                "source", "number_flights_matched",
            ]
        )

    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    for col in ["airline", "origin", "destination", "airport_pair", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    for col in ["year", "month", "number_flights_matched"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["avg_arrival_delay_min", "pct_flights_15min_late", "cancelled_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def has_real_european_context(path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH) -> bool:
    path = Path(path)
    return path.exists() and not load_european_context(path).empty


def _row_to_context(row: pd.Series, matched_level: str) -> EuropeanPunctualityContext:
    return EuropeanPunctualityContext(
        status="matched",
        source=str(row.get("source", "unknown")),
        year=None if pd.isna(row.get("year")) else int(row.get("year")),
        month=None if pd.isna(row.get("month")) else int(row.get("month")),
        airline=str(row.get("airline")) if "airline" in row else None,
        origin=str(row.get("origin")) if "origin" in row else None,
        destination=str(row.get("destination")) if "destination" in row else None,
        airport_pair=str(row.get("airport_pair")) if "airport_pair" in row else None,
        avg_arrival_delay_min=None if pd.isna(row.get("avg_arrival_delay_min")) else float(row.get("avg_arrival_delay_min")),
        pct_flights_15min_late=None if pd.isna(row.get("pct_flights_15min_late")) else float(row.get("pct_flights_15min_late")),
        cancelled_pct=None if pd.isna(row.get("cancelled_pct")) else float(row.get("cancelled_pct")),
        number_flights_matched=None if pd.isna(row.get("number_flights_matched")) else int(row.get("number_flights_matched")),
        matched_level=matched_level,
    )


def _weighted_average(df: pd.DataFrame, value_col: str) -> float | None:
    if value_col not in df.columns or df[value_col].dropna().empty:
        return None
    if "number_flights_matched" in df.columns and df["number_flights_matched"].fillna(0).sum() > 0:
        tmp = df[[value_col, "number_flights_matched"]].dropna()
        if not tmp.empty and tmp["number_flights_matched"].sum() > 0:
            return float((tmp[value_col] * tmp["number_flights_matched"]).sum() / tmp["number_flights_matched"].sum())
    return float(df[value_col].mean())


def _aggregate_to_context(df: pd.DataFrame, matched_level: str, source: str, airline: str, origin: str, destination: str, month: int) -> EuropeanPunctualityContext:
    return EuropeanPunctualityContext(
        status="matched",
        source=source,
        year=None if "year" not in df.columns or df["year"].dropna().empty else int(df["year"].dropna().max()),
        month=month,
        airline=airline,
        origin=origin,
        destination=destination,
        airport_pair=f"{origin}-{destination}",
        avg_arrival_delay_min=_weighted_average(df, "avg_arrival_delay_min"),
        pct_flights_15min_late=_weighted_average(df, "pct_flights_15min_late"),
        cancelled_pct=_weighted_average(df, "cancelled_pct"),
        number_flights_matched=None if "number_flights_matched" not in df.columns else int(df["number_flights_matched"].fillna(0).sum()),
        matched_level=matched_level,
    )


def lookup_european_context(airline: str, origin: str, destination: str, month: int, context_path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH) -> EuropeanPunctualityContext:
    df = load_european_context(context_path)
    airline = airline.upper(); origin = origin.upper(); destination = destination.upper()
    pair = f"{origin}-{destination}"
    if df.empty:
        return EuropeanPunctualityContext(
            status="unavailable", source=str(resolve_context_path(context_path)), airline=airline,
            origin=origin, destination=destination, month=month, airport_pair=pair, matched_level="none"
        )

    exact = df[(df["airline"] == airline) & (df["origin"] == origin) & (df["destination"] == destination) & (df["month"] == month)]
    if not exact.empty:
        return _aggregate_to_context(exact, "airline_route_month", "OFFICIAL_EUROPE_CONTEXT", airline, origin, destination, month)

    route_month = df[(df["origin"] == origin) & (df["destination"] == destination) & (df["month"] == month)]
    if not route_month.empty:
        return _aggregate_to_context(route_month, "route_month", "OFFICIAL_EUROPE_CONTEXT_ROUTE_MONTH", airline, origin, destination, month)

    route_any_month = df[(df["origin"] == origin) & (df["destination"] == destination)]
    if not route_any_month.empty:
        return _aggregate_to_context(route_any_month, "route_average", "OFFICIAL_EUROPE_CONTEXT_ROUTE_AVERAGE", airline, origin, destination, month)

    return EuropeanPunctualityContext(
        status="missing", source=str(resolve_context_path(context_path)), airline=airline, origin=origin,
        destination=destination, month=month, airport_pair=pair, matched_level="none"
    )


def summarize_european_context(path: str | Path = DEFAULT_EUROPE_CONTEXT_PATH) -> dict:
    resolved_path = resolve_context_path(path)
    df = load_european_context(path)
    if df.empty:
        return {
            "available": False, "real_data": False, "rows": 0, "source_path": str(resolved_path),
            "is_sample": False, "airlines": [], "routes": [], "months": [],
            "note": "No real European context found. Run scripts.download_uk_caa_punctuality and scripts.prepare_uk_caa_context.",
        }

    routes = sorted({f"{r.origin}-{r.destination}" for r in df.itertuples()})
    is_sample = resolved_path.name == SAMPLE_EUROPE_CONTEXT_PATH.name
    return {
        "available": True,
        "real_data": not is_sample,
        "rows": int(len(df)),
        "source_path": str(resolved_path),
        "is_sample": is_sample,
        "airlines": sorted(df["airline"].dropna().unique().tolist()),
        "routes": routes,
        "months": sorted([int(x) for x in df["month"].dropna().unique().tolist()]),
        "total_matched_flights": None if "number_flights_matched" not in df.columns else int(df["number_flights_matched"].fillna(0).sum()),
        "note": "Real/generated European punctuality context." if not is_sample else "Sample fallback enabled by environment variable.",
    }
