"""Target-free schedule density features shared by training and serving."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.features.build_features import hhmm_to_minute
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _key(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def _mean_map(frame: pd.DataFrame, group_cols: list[str], value: str) -> dict[str, float]:
    grouped = frame.groupby(group_cols, observed=True)[value].mean()
    return {_key(*idx if isinstance(idx, tuple) else (idx,)): float(v) for idx, v in grouped.items()}


def _count_map(frame: pd.DataFrame, group_cols: list[str]) -> dict[str, float]:
    grouped = frame.groupby(group_cols, observed=True).size()
    return {_key(*idx if isinstance(idx, tuple) else (idx,)): float(v) for idx, v in grouped.items()}


@dataclass
class ScheduleContextReference:
    """Compact timetable lookup maps that never use delay outcomes."""

    origin_slot_counts: dict[str, float] = field(default_factory=dict)
    dest_slot_counts: dict[str, float] = field(default_factory=dict)
    origin_daily: dict[str, float] = field(default_factory=dict)
    dest_daily: dict[str, float] = field(default_factory=dict)
    carrier_origin_daily: dict[str, float] = field(default_factory=dict)
    route_daily: dict[str, float] = field(default_factory=dict)
    fitted_rows: int = 0
    date_start: str | None = None
    date_end: str | None = None

    def fit(self, df: pd.DataFrame) -> "ScheduleContextReference":
        required = {"FlightDate", "DayOfWeek", "Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Schedule context missing columns: {sorted(missing)}")
        frame = df[list(required)].copy()
        frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed").dt.normalize()
        for col in ("Airline", "Origin", "Dest"):
            frame[col] = frame[col].astype(str).str.upper().str.strip()
        frame["DayOfWeek"] = pd.to_numeric(frame["DayOfWeek"], errors="raise").astype(int)
        frame["DepSlot"] = (hhmm_to_minute(frame["CRSDepTime"]) // 30).astype(int)
        frame["ArrSlot"] = (hhmm_to_minute(frame["CRSArrTime"]) // 30).astype(int)
        frame["Route"] = frame["Origin"] + "_" + frame["Dest"]

        weekday_occurrences = (
            frame[["FlightDate", "DayOfWeek"]]
            .drop_duplicates()
            .groupby("DayOfWeek", observed=True)
            .size()
            .astype(float)
            .to_dict()
        )

        def typical_day_count(group_cols: list[str]) -> dict[str, float]:
            grouped = frame.groupby(group_cols, observed=True).size()
            result: dict[str, float] = {}
            for index, value in grouped.items():
                parts = index if isinstance(index, tuple) else (index,)
                denominator = weekday_occurrences.get(int(parts[0]), 52.18)
                result[_key(*parts)] = float(value / max(denominator, 1.0))
            return result

        self.origin_slot_counts = typical_day_count(["DayOfWeek", "Origin", "DepSlot"])
        self.dest_slot_counts = typical_day_count(["DayOfWeek", "Dest", "ArrSlot"])
        self.origin_daily = typical_day_count(["DayOfWeek", "Origin"])
        self.dest_daily = typical_day_count(["DayOfWeek", "Dest"])
        self.carrier_origin_daily = typical_day_count(["DayOfWeek", "Airline", "Origin"])
        self.route_daily = typical_day_count(["DayOfWeek", "Route"])
        self.fitted_rows = len(frame)
        self.date_start = str(frame["FlightDate"].min().date())
        self.date_end = str(frame["FlightDate"].max().date())
        logger.info(
            "Fit target-free schedule context on %d rows (%s to %s)",
            self.fitted_rows, self.date_start, self.date_end,
        )
        return self

    @staticmethod
    def _lookup(mapping: dict[str, float], keys: pd.Series) -> pd.Series:
        return keys.map(mapping).fillna(0.0).astype(float)

    @staticmethod
    def _window(mapping: dict[str, float], dow: pd.Series, airport: pd.Series, slot: pd.Series, radius: int) -> pd.Series:
        output = np.zeros(len(slot), dtype=float)
        for offset in range(-radius, radius + 1):
            shifted = (slot + offset) % 48
            keys = pd.Series(
                [_key(d, a, s) for d, a, s in zip(dow, airport, shifted, strict=False)],
                index=slot.index,
            )
            output += keys.map(mapping).fillna(0.0).to_numpy(dtype=float)
        return pd.Series(output, index=slot.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"DayOfWeek", "Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime"}
        if not required.issubset(df.columns):
            return df.copy()
        out = df.copy()
        dow = pd.to_numeric(out["DayOfWeek"], errors="coerce").fillna(1).astype(int)
        airline = out["Airline"].astype(str).str.upper().str.strip()
        origin = out["Origin"].astype(str).str.upper().str.strip()
        dest = out["Dest"].astype(str).str.upper().str.strip()
        route = origin + "_" + dest
        dep_slot = (hhmm_to_minute(out["CRSDepTime"]) // 30).astype(int)
        arr_slot = (hhmm_to_minute(out["CRSArrTime"]) // 30).astype(int)

        out["OriginScheduledDepartures30m"] = self._window(self.origin_slot_counts, dow, origin, dep_slot, 1)
        out["OriginScheduledDepartures60m"] = self._window(self.origin_slot_counts, dow, origin, dep_slot, 2)
        out["OriginScheduledDepartures120m"] = self._window(self.origin_slot_counts, dow, origin, dep_slot, 4)
        out["DestScheduledArrivals30m"] = self._window(self.dest_slot_counts, dow, dest, arr_slot, 1)
        out["DestScheduledArrivals60m"] = self._window(self.dest_slot_counts, dow, dest, arr_slot, 2)
        out["DestScheduledArrivals120m"] = self._window(self.dest_slot_counts, dow, dest, arr_slot, 4)

        origin_keys = pd.Series([_key(d, a) for d, a in zip(dow, origin, strict=False)], index=out.index)
        dest_keys = pd.Series([_key(d, a) for d, a in zip(dow, dest, strict=False)], index=out.index)
        carrier_origin_keys = pd.Series([_key(d, c, a) for d, c, a in zip(dow, airline, origin, strict=False)], index=out.index)
        route_keys = pd.Series([_key(d, r) for d, r in zip(dow, route, strict=False)], index=out.index)
        out["OriginDailyScheduledFlights"] = self._lookup(self.origin_daily, origin_keys)
        out["DestDailyScheduledFlights"] = self._lookup(self.dest_daily, dest_keys)
        out["CarrierOriginDailyScheduledFlights"] = self._lookup(self.carrier_origin_daily, carrier_origin_keys)
        out["RouteDailyScheduledFlights"] = self._lookup(self.route_daily, route_keys)
        out["OriginBankShare60m"] = (
            out["OriginScheduledDepartures60m"] / out["OriginDailyScheduledFlights"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)
        out["DestBankShare60m"] = (
            out["DestScheduledArrivals60m"] / out["DestDailyScheduledFlights"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)
        return out

    def compatible_with(self, df: pd.DataFrame) -> bool:
        if not self.fitted_rows or "FlightDate" not in df.columns:
            return False
        dates = pd.to_datetime(df["FlightDate"], errors="coerce", format="mixed")
        return (
            len(df) == self.fitted_rows
            and str(dates.min().date()) == self.date_start
            and str(dates.max().date()) == self.date_end
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_slot_counts": self.origin_slot_counts,
            "dest_slot_counts": self.dest_slot_counts,
            "origin_daily": self.origin_daily,
            "dest_daily": self.dest_daily,
            "carrier_origin_daily": self.carrier_origin_daily,
            "route_daily": self.route_daily,
            "fitted_rows": self.fitted_rows,
            "date_start": self.date_start,
            "date_end": self.date_end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScheduleContextReference | None":
        if not data:
            return None
        return cls(**data)
