"""Leakage-aware historical, support, recency and schedule-context features."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import GLOBAL_FALLBACK_DELAY_RATE, SCHEDULE_CONGESTION_FEATURES, TARGET_COL
from src.features.schedule_context import ScheduleContextReference
from src.utils.logging import get_logger

logger = get_logger(__name__)

RATE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Airline", "CarrierDelayRate", "carrier_rates", "carrier_counts"),
    ("Route", "RouteDelayRate", "route_rates", "route_counts"),
    ("Origin", "OriginDelayRate", "origin_rates", "origin_counts"),
    ("Dest", "DestDelayRate", "dest_rates", "dest_counts"),
    ("CarrierRoute", "CarrierRouteDelayRate", "carrier_route_rates", "carrier_route_counts"),
    ("AirlineOrigin", "AirlineOriginDelayRate", "airline_origin_rates", "airline_origin_counts"),
    ("AirlineDest", "AirlineDestDelayRate", "airline_dest_rates", "airline_dest_counts"),
    ("OriginDepHour", "OriginHourDelayRate", "origin_hour_rates", "origin_hour_counts"),
    ("DestArrHour", "DestHourDelayRate", "dest_hour_rates", "dest_hour_counts"),
    ("CarrierDepHour", "CarrierDepHourDelayRate", "carrier_dep_hour_rates", "carrier_dep_hour_counts"),
)
SHARE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Route", "RouteFlightShare", "route_shares"),
    ("CarrierRoute", "CarrierRouteFlightShare", "carrier_route_shares"),
    ("AirlineOrigin", "AirlineOriginFlightShare", "airline_origin_shares"),
    ("OriginDepHour", "OriginHourFlightShare", "origin_hour_shares"),
    ("DestArrHour", "DestHourFlightShare", "dest_hour_shares"),
    ("CarrierDepHour", "CarrierDepHourFlightShare", "carrier_dep_hour_shares"),
)
RECENCY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Airline", "CarrierDelay", "CarrierDelayRate"),
    ("Route", "RouteDelay", "RouteDelayRate"),
    ("Origin", "OriginDelay", "OriginDelayRate"),
    ("Dest", "DestDelay", "DestDelayRate"),
)


def _ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    definitions = {
        "CarrierRoute": ("Airline", "Route"),
        "AirlineOrigin": ("Airline", "Origin"),
        "AirlineDest": ("Airline", "Dest"),
        "OriginDepHour": ("Origin", "DepHour"),
        "DestArrHour": ("Dest", "ArrHour"),
        "CarrierDepHour": ("Airline", "DepHour"),
    }
    for name, (left, right) in definitions.items():
        if name not in out.columns and {left, right}.issubset(out.columns):
            out[name] = out[left].astype(str) + "_" + out[right].astype(str)
    return out


def _map(df: pd.DataFrame, key: str, mapping: dict, fallback: float) -> pd.Series:
    if key not in df.columns:
        return pd.Series(fallback, index=df.index, dtype=float)
    return df[key].map(mapping).fillna(fallback).astype(float)


def _rate_count(df: pd.DataFrame, key: str, global_rate: float, alpha: float):
    if key not in df.columns:
        return {}, {}
    grouped = df.groupby(key, observed=True)[TARGET_COL].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * global_rate) / (grouped["count"] + alpha)
    return rates.astype(float).to_dict(), grouped["count"].astype(int).to_dict()


def _count_feature(rate_feature: str) -> str:
    return rate_feature.replace("DelayRate", "HistoryCount")


def _smoothed(total: float, count: float, prior: float, alpha: float) -> float:
    return float((total + alpha * prior) / (count + alpha)) if count + alpha else float(prior)


@dataclass
class HistoricalAggregates:
    DEFAULT_SMOOTHING_STRENGTH: ClassVar[float] = 50.0
    RECENCY_WINDOWS: ClassVar[tuple[int, int]] = (28, 90)
    EWMA_HALF_LIFE_DAYS: ClassVar[float] = 28.0

    carrier_rates: dict[str, float] = field(default_factory=dict)
    route_rates: dict[str, float] = field(default_factory=dict)
    origin_rates: dict[str, float] = field(default_factory=dict)
    dest_rates: dict[str, float] = field(default_factory=dict)
    carrier_route_rates: dict[str, float] = field(default_factory=dict)
    airline_origin_rates: dict[str, float] = field(default_factory=dict)
    airline_dest_rates: dict[str, float] = field(default_factory=dict)
    origin_hour_rates: dict[str, float] = field(default_factory=dict)
    dest_hour_rates: dict[str, float] = field(default_factory=dict)
    carrier_dep_hour_rates: dict[str, float] = field(default_factory=dict)

    carrier_counts: dict[str, int] = field(default_factory=dict)
    route_counts: dict[str, int] = field(default_factory=dict)
    origin_counts: dict[str, int] = field(default_factory=dict)
    dest_counts: dict[str, int] = field(default_factory=dict)
    carrier_route_counts: dict[str, int] = field(default_factory=dict)
    airline_origin_counts: dict[str, int] = field(default_factory=dict)
    airline_dest_counts: dict[str, int] = field(default_factory=dict)
    origin_hour_counts: dict[str, int] = field(default_factory=dict)
    dest_hour_counts: dict[str, int] = field(default_factory=dict)
    carrier_dep_hour_counts: dict[str, int] = field(default_factory=dict)

    route_shares: dict[str, float] = field(default_factory=dict)
    carrier_route_shares: dict[str, float] = field(default_factory=dict)
    airline_origin_shares: dict[str, float] = field(default_factory=dict)
    origin_hour_shares: dict[str, float] = field(default_factory=dict)
    dest_hour_shares: dict[str, float] = field(default_factory=dict)
    carrier_dep_hour_shares: dict[str, float] = field(default_factory=dict)
    recency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    schedule_context: ScheduleContextReference | None = None

    global_fallback: float = GLOBAL_FALLBACK_DELAY_RATE
    frequency_fallback: float = 0.0
    smoothing_strength: float = DEFAULT_SMOOTHING_STRENGTH
    ordered_encoding: bool = False
    fitted_cutoff_date: str | None = None

    def _with_context(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.schedule_context.transform(df) if self.schedule_context is not None else df.copy()
        for column in SCHEDULE_CONGESTION_FEATURES:
            if column not in out.columns:
                out[column] = 0.0
        return out

    def fit(self, train_df: pd.DataFrame) -> "HistoricalAggregates":
        frame = _ensure_keys(self._with_context(train_df))
        if TARGET_COL not in frame.columns:
            raise KeyError(f"Expected target column {TARGET_COL!r}")
        self.global_fallback = float(frame[TARGET_COL].mean())
        for key, _, rate_attr, count_attr in RATE_SPECS:
            rates, counts = _rate_count(frame, key, self.global_fallback, self.smoothing_strength)
            setattr(self, rate_attr, rates)
            setattr(self, count_attr, counts)
        total = max(len(frame), 1)
        count_attrs = {key: count_attr for key, _, _, count_attr in RATE_SPECS}
        for key, _, share_attr in SHARE_SPECS:
            counts = getattr(self, count_attrs[key])
            setattr(self, share_attr, {k: float(v / total) for k, v in counts.items()})
        if "FlightDate" in frame.columns:
            dates = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed").dt.normalize()
            self.fitted_cutoff_date = str(dates.max().date())
            self._fit_recency(frame.assign(__history_date=dates))
        logger.info(
            "Fit historical aggregates: %d routes, alpha=%.1f, cutoff=%s",
            len(self.route_rates), self.smoothing_strength, self.fitted_cutoff_date,
        )
        return self

    def _fit_recency(self, frame: pd.DataFrame) -> None:
        cutoff = frame["__history_date"].max() + pd.Timedelta(days=1)
        decay_base = 0.5 ** (1.0 / self.EWMA_HALF_LIFE_DAYS)
        for key, prefix, _ in RECENCY_SPECS:
            if key not in frame.columns:
                continue
            daily = frame.groupby([key, "__history_date"], observed=True)[TARGET_COL].agg(["sum", "count"]).reset_index()
            for window in self.RECENCY_WINDOWS:
                recent = daily[daily["__history_date"] >= cutoff - pd.Timedelta(days=window)]
                grouped = recent.groupby(key, observed=True)[["sum", "count"]].sum()
                self.recency_maps[f"{prefix}Rate{window}d"] = {
                    str(k): _smoothed(row["sum"], row["count"], self.global_fallback, self.smoothing_strength)
                    for k, row in grouped.iterrows()
                }
            ewma: dict[str, float] = {}
            for group_key, group in daily.groupby(key, observed=True):
                weighted_sum = 0.0
                weighted_count = 0.0
                last = None
                for _, date, target_sum, target_count in group.sort_values("__history_date").itertuples(index=False, name=None):
                    if last is not None:
                        decay = decay_base ** max((date - last).days, 0)
                        weighted_sum *= decay
                        weighted_count *= decay
                    weighted_sum += float(target_sum)
                    weighted_count += float(target_count)
                    last = date
                if last is not None:
                    decay = decay_base ** max((cutoff - last).days, 0)
                    weighted_sum *= decay
                    weighted_count *= decay
                ewma[str(group_key)] = _smoothed(weighted_sum, weighted_count, self.global_fallback, self.smoothing_strength)
            self.recency_maps[f"{prefix}RateEWMA"] = ewma

    def fit_transform_ordered(self, train_df: pd.DataFrame, *, time_col: str = "FlightDate") -> pd.DataFrame:
        frame = _ensure_keys(self._with_context(train_df))
        if TARGET_COL not in frame.columns:
            raise KeyError(f"Expected target column {TARGET_COL!r}")
        if time_col not in frame.columns:
            encoded = self._fit_transform_leave_one_out(frame)
            self.fit(frame)
            return encoded
        dates = pd.to_datetime(frame[time_col], errors="raise", format="mixed").dt.normalize()
        out = frame.copy()
        out["__history_date"] = dates.to_numpy()
        date_stats = out.groupby("__history_date", observed=True)[TARGET_COL].agg(["sum", "count"]).sort_index()
        date_stats["prior_sum"] = date_stats["sum"].cumsum() - date_stats["sum"]
        date_stats["prior_count"] = date_stats["count"].cumsum() - date_stats["count"]
        date_stats["prior_global_rate"] = np.where(
            date_stats["prior_count"] > 0,
            date_stats["prior_sum"] / date_stats["prior_count"],
            GLOBAL_FALLBACK_DELAY_RATE,
        )
        global_rate_by_date = date_stats["prior_global_rate"]
        global_count_by_date = date_stats["prior_count"]
        row_global_rate = out["__history_date"].map(global_rate_by_date).astype(float)
        row_global_count = out["__history_date"].map(global_count_by_date).fillna(0).astype(float)

        ordered_counts: dict[str, pd.Series] = {}
        for key, feature, _, _ in RATE_SPECS:
            if key not in out.columns:
                out[feature] = row_global_rate
                count = pd.Series(0.0, index=out.index)
            else:
                stats = out.groupby([key, "__history_date"], observed=True)[TARGET_COL].agg(["sum", "count"]).reset_index().sort_values([key, "__history_date"])
                stats["prior_sum"] = stats.groupby(key, observed=True)["sum"].cumsum() - stats["sum"]
                stats["prior_count"] = stats.groupby(key, observed=True)["count"].cumsum() - stats["count"]
                stats["prior_global_rate"] = stats["__history_date"].map(global_rate_by_date)
                denominator = stats["prior_count"] + self.smoothing_strength
                numerator = (
                    stats["prior_sum"]
                    + self.smoothing_strength * stats["prior_global_rate"]
                )
                stats[feature] = np.where(
                    denominator > 0,
                    numerator / denominator,
                    stats["prior_global_rate"],
                )
                lookup = pd.MultiIndex.from_frame(out[[key, "__history_date"]])
                index = pd.MultiIndex.from_frame(stats[[key, "__history_date"]])
                out[feature] = pd.Series(stats[feature].to_numpy(), index=index).reindex(lookup).to_numpy(dtype=float)
                count = pd.Series(pd.Series(stats["prior_count"].to_numpy(), index=index).reindex(lookup).to_numpy(dtype=float), index=out.index).fillna(0.0)
            ordered_counts[key] = count
            count_name = _count_feature(feature)
            out[count_name] = count
            out[count_name.replace("Count", "LogCount")] = np.log1p(count)

        for key, feature, _ in SHARE_SPECS:
            denominator = row_global_count.replace(0, np.nan)
            out[feature] = (ordered_counts.get(key, pd.Series(0.0, index=out.index)) / denominator).fillna(0.0)
        self._ordered_recency(out, global_rate_by_date)
        out = out.drop(columns=["__history_date"])
        self.fit(frame)
        self.ordered_encoding = True
        return out

    def _ordered_recency(self, out: pd.DataFrame, global_rate_by_date: pd.Series) -> None:
        alpha = self.smoothing_strength
        decay_base = 0.5 ** (1.0 / self.EWMA_HALF_LIFE_DAYS)
        for key, prefix, long_feature in RECENCY_SPECS:
            for suffix in ("Rate28d", "Rate90d", "RateEWMA"):
                out[f"{prefix}{suffix}"] = out["__history_date"].map(global_rate_by_date).astype(float)
            if key not in out.columns:
                out[f"{prefix}Trend28d"] = 0.0
                continue
            daily = out.groupby([key, "__history_date"], observed=True)[TARGET_COL].agg(["sum", "count"]).reset_index().sort_values([key, "__history_date"])
            records: list[tuple[object, pd.Timestamp, float, float, float]] = []
            for group_key, group in daily.groupby(key, observed=True):
                q28: deque = deque()
                q90: deque = deque()
                sum28 = count28 = sum90 = count90 = 0.0
                ew_sum = ew_count = 0.0
                last = None
                for _, date, target_sum, target_count in group.itertuples(index=False, name=None):
                    while q28 and q28[0][0] < date - pd.Timedelta(days=28):
                        _, s, c = q28.popleft()
                        sum28 -= s
                        count28 -= c
                    while q90 and q90[0][0] < date - pd.Timedelta(days=90):
                        _, s, c = q90.popleft()
                        sum90 -= s
                        count90 -= c
                    if last is not None:
                        decay = decay_base ** max((date - last).days, 0)
                        ew_sum *= decay
                        ew_count *= decay
                    prior = float(global_rate_by_date.get(date, GLOBAL_FALLBACK_DELAY_RATE))
                    records.append((group_key, date,
                        _smoothed(sum28, count28, prior, alpha),
                        _smoothed(sum90, count90, prior, alpha),
                        _smoothed(ew_sum, ew_count, prior, alpha)))
                    values = (date, float(target_sum), float(target_count))
                    q28.append(values)
                    q90.append(values)
                    sum28 += values[1]
                    count28 += values[2]
                    sum90 += values[1]
                    count90 += values[2]
                    ew_sum += values[1]
                    ew_count += values[2]
                    last = date
            lookup_table = pd.DataFrame(records, columns=[key, "__history_date", "r28", "r90", "ewma"])
            lookup = pd.MultiIndex.from_frame(out[[key, "__history_date"]])
            index = pd.MultiIndex.from_frame(lookup_table[[key, "__history_date"]])
            for column, suffix in (("r28", "Rate28d"), ("r90", "Rate90d"), ("ewma", "RateEWMA")):
                out[f"{prefix}{suffix}"] = pd.Series(lookup_table[column].to_numpy(), index=index).reindex(lookup).to_numpy(dtype=float)
            out[f"{prefix}Trend28d"] = out[f"{prefix}Rate28d"] - out[long_feature]

    def _fit_transform_leave_one_out(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        global_rate = float(out[TARGET_COL].mean())
        n_total = max(len(out) - 1, 1)
        for key, feature, _, _ in RATE_SPECS:
            if key in out.columns:
                sums = out.groupby(key, observed=True)[TARGET_COL].transform("sum") - out[TARGET_COL]
                counts = out.groupby(key, observed=True)[TARGET_COL].transform("count") - 1
                denominator = counts + self.smoothing_strength
                numerator = sums + self.smoothing_strength * global_rate
                out[feature] = np.where(
                    denominator > 0,
                    numerator / denominator,
                    global_rate,
                )
            else:
                counts = pd.Series(0, index=out.index)
                out[feature] = global_rate
            count_name = _count_feature(feature)
            out[count_name] = counts
            out[count_name.replace("Count", "LogCount")] = np.log1p(counts.clip(lower=0))
        for key, feature, _ in SHARE_SPECS:
            out[feature] = out.groupby(key, observed=True)[TARGET_COL].transform("count").sub(1).clip(lower=0) / n_total if key in out.columns else 0.0
        for _, prefix, long_feature in RECENCY_SPECS:
            out[f"{prefix}Rate28d"] = out[long_feature]
            out[f"{prefix}Rate90d"] = out[long_feature]
            out[f"{prefix}RateEWMA"] = out[long_feature]
            out[f"{prefix}Trend28d"] = 0.0
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = _ensure_keys(self._with_context(df))
        for key, feature, rate_attr, count_attr in RATE_SPECS:
            out[feature] = _map(out, key, getattr(self, rate_attr), self.global_fallback)
            count_name = _count_feature(feature)
            out[count_name] = _map(out, key, getattr(self, count_attr), 0.0)
            out[count_name.replace("Count", "LogCount")] = np.log1p(out[count_name])
        for key, feature, share_attr in SHARE_SPECS:
            out[feature] = _map(out, key, getattr(self, share_attr), self.frequency_fallback)
        long_lookup = {key: long_feature for key, _, long_feature in RECENCY_SPECS}
        for key, prefix, long_feature in RECENCY_SPECS:
            for suffix in ("Rate28d", "Rate90d", "RateEWMA"):
                out[f"{prefix}{suffix}"] = _map(out, key, self.recency_maps.get(f"{prefix}{suffix}", {}), self.global_fallback)
            out[f"{prefix}Trend28d"] = out[f"{prefix}Rate28d"] - out[long_lookup[key]]
        return out

    def lookup_single(self, airline: str, origin: str, dest: str, route: str) -> dict[str, float]:
        return {
            "CarrierDelayRate": self.carrier_rates.get(airline, self.global_fallback),
            "RouteDelayRate": self.route_rates.get(route, self.global_fallback),
            "OriginDelayRate": self.origin_rates.get(origin, self.global_fallback),
            "DestDelayRate": self.dest_rates.get(dest, self.global_fallback),
        }

    def to_dict(self) -> dict:
        fields = {
            "global_fallback": self.global_fallback,
            "frequency_fallback": self.frequency_fallback,
            "smoothing_strength": self.smoothing_strength,
            "ordered_encoding": self.ordered_encoding,
            "fitted_cutoff_date": self.fitted_cutoff_date,
            "recency_maps": self.recency_maps,
            "schedule_context": self.schedule_context.to_dict() if self.schedule_context else None,
        }
        for _, _, rate_attr, count_attr in RATE_SPECS:
            fields[rate_attr] = getattr(self, rate_attr)
            fields[count_attr] = getattr(self, count_attr)
        for _, _, share_attr in SHARE_SPECS:
            fields[share_attr] = getattr(self, share_attr)
        return fields

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalAggregates":
        kwargs = {
            "global_fallback": float(data.get("global_fallback", GLOBAL_FALLBACK_DELAY_RATE)),
            "frequency_fallback": float(data.get("frequency_fallback", 0.0)),
            "smoothing_strength": float(data.get("smoothing_strength", cls.DEFAULT_SMOOTHING_STRENGTH)),
            "ordered_encoding": bool(data.get("ordered_encoding", False)),
            "fitted_cutoff_date": data.get("fitted_cutoff_date"),
            "recency_maps": data.get("recency_maps", {}),
            "schedule_context": ScheduleContextReference.from_dict(data.get("schedule_context")),
        }
        for _, _, rate_attr, count_attr in RATE_SPECS:
            kwargs[rate_attr] = data.get(rate_attr, {})
            kwargs[count_attr] = {str(k): int(v) for k, v in data.get(count_attr, {}).items()}
        for _, _, share_attr in SHARE_SPECS:
            kwargs[share_attr] = data.get(share_attr, {})
        return cls(**kwargs)
