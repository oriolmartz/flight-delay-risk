"""Leakage-aware historical cohort features.

The public artifact stores smoothed cohort rates fitted on the model-training
period. During model fitting, :meth:`fit_transform_ordered` creates each
training row from history strictly earlier than its ``FlightDate``. Rows on the
same date therefore never contribute targets to one another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from src.config import GLOBAL_FALLBACK_DELAY_RATE, TARGET_COL
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


def _ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "CarrierRoute" not in df.columns and {"Airline", "Route"}.issubset(df.columns):
        df["CarrierRoute"] = df["Airline"].astype(str) + "_" + df["Route"].astype(str)
    if "AirlineOrigin" not in df.columns and {"Airline", "Origin"}.issubset(df.columns):
        df["AirlineOrigin"] = df["Airline"].astype(str) + "_" + df["Origin"].astype(str)
    if "AirlineDest" not in df.columns and {"Airline", "Dest"}.issubset(df.columns):
        df["AirlineDest"] = df["Airline"].astype(str) + "_" + df["Dest"].astype(str)
    if "OriginDepHour" not in df.columns and {"Origin", "DepHour"}.issubset(df.columns):
        df["OriginDepHour"] = df["Origin"].astype(str) + "_" + df["DepHour"].astype(str)
    if "DestArrHour" not in df.columns and {"Dest", "ArrHour"}.issubset(df.columns):
        df["DestArrHour"] = df["Dest"].astype(str) + "_" + df["ArrHour"].astype(str)
    if "CarrierDepHour" not in df.columns and {"Airline", "DepHour"}.issubset(df.columns):
        df["CarrierDepHour"] = df["Airline"].astype(str) + "_" + df["DepHour"].astype(str)
    return df


def _map_with_fallback(
    df: pd.DataFrame,
    key: str,
    mapping: dict[str, float] | dict[str, int],
    fallback: float,
) -> pd.Series:
    if key not in df.columns:
        return pd.Series(fallback, index=df.index, dtype=float)
    return df[key].map(mapping).fillna(fallback)


def _smoothed_rate_and_count(
    df: pd.DataFrame,
    key: str,
    global_rate: float,
    smoothing_strength: float,
) -> tuple[dict[str, float], dict[str, int]]:
    if key not in df.columns:
        return {}, {}
    grouped = df.groupby(key, observed=True)[TARGET_COL].agg(["sum", "count"])
    rates = (grouped["sum"] + smoothing_strength * global_rate) / (
        grouped["count"] + smoothing_strength
    )
    return rates.astype(float).to_dict(), grouped["count"].astype(int).to_dict()


def _share_map_from_counts(counts: dict[str, int], total: int) -> dict[str, float]:
    denominator = max(int(total), 1)
    return {key: float(value / denominator) for key, value in counts.items()}


@dataclass
class HistoricalAggregates:
    """Smoothed delay rates, exact support counts and frequency shares."""

    DEFAULT_SMOOTHING_STRENGTH: ClassVar[float] = 50.0

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

    global_fallback: float = GLOBAL_FALLBACK_DELAY_RATE
    frequency_fallback: float = 0.0
    smoothing_strength: float = DEFAULT_SMOOTHING_STRENGTH
    ordered_encoding: bool = False

    def fit(self, train_df: pd.DataFrame) -> "HistoricalAggregates":
        """Fit smoothed maps on a historical training period."""
        train_df = _ensure_keys(train_df)
        if TARGET_COL not in train_df.columns:
            raise KeyError(f"Expected target column {TARGET_COL!r}")
        self.global_fallback = float(train_df[TARGET_COL].mean())

        for key, _, rate_attr, count_attr in RATE_SPECS:
            rates, counts = _smoothed_rate_and_count(
                train_df, key, self.global_fallback, self.smoothing_strength
            )
            setattr(self, rate_attr, rates)
            setattr(self, count_attr, counts)

        total = len(train_df)
        count_lookup = {key: count_attr for key, _, _, count_attr in RATE_SPECS}
        for key, _, share_attr in SHARE_SPECS:
            setattr(
                self,
                share_attr,
                _share_map_from_counts(getattr(self, count_lookup[key]), total),
            )

        logger.info(
            "Fit smoothed historical aggregates: %d routes, %d carrier-routes, "
            "alpha=%.1f, global fallback=%.4f",
            len(self.route_rates),
            len(self.carrier_route_rates),
            self.smoothing_strength,
            self.global_fallback,
        )
        return self

    def fit_transform_ordered(
        self,
        train_df: pd.DataFrame,
        *,
        time_col: str = "FlightDate",
    ) -> pd.DataFrame:
        """Create training features using targets from strictly earlier dates.

        Encoding is calculated at date granularity, not row granularity. This
        prevents flights on the same date from contributing labels to one
        another and removes self-target leakage from historical-rate features.
        The instance is then fitted on the complete training period for use on
        future validation, test and inference rows.
        """
        frame = _ensure_keys(train_df)
        if TARGET_COL not in frame.columns:
            raise KeyError(f"Expected target column {TARGET_COL!r}")
        if time_col not in frame.columns:
            logger.warning(
                "%s missing; using leave-one-out historical encoding for training only",
                time_col,
            )
            encoded = self._fit_transform_leave_one_out(frame)
            self.fit(frame)
            self.ordered_encoding = False
            return encoded

        dates = pd.to_datetime(frame[time_col], errors="coerce", format="mixed").dt.normalize()
        if dates.isna().any():
            raise ValueError(f"{time_col} contains unparseable values")

        out = frame.copy()
        out["__history_date"] = dates.to_numpy()

        date_stats = (
            out.groupby("__history_date", observed=True)[TARGET_COL]
            .agg(["sum", "count"])
            .sort_index()
        )
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

        ordered_count_by_key: dict[str, pd.Series] = {}
        for key, feature, _, _ in RATE_SPECS:
            if key not in out.columns:
                out[feature] = row_global_rate
                continue

            stats = (
                out.groupby([key, "__history_date"], observed=True)[TARGET_COL]
                .agg(["sum", "count"])
                .reset_index()
                .sort_values([key, "__history_date"])
            )
            stats["prior_sum"] = stats.groupby(key, observed=True)["sum"].cumsum() - stats["sum"]
            stats["prior_count"] = (
                stats.groupby(key, observed=True)["count"].cumsum() - stats["count"]
            )
            stats["prior_global_rate"] = stats["__history_date"].map(global_rate_by_date)
            denominator = stats["prior_count"] + self.smoothing_strength
            numerator = (
                stats["prior_sum"] + self.smoothing_strength * stats["prior_global_rate"]
            )
            stats[feature] = np.where(
                denominator > 0,
                numerator / denominator,
                stats["prior_global_rate"],
            )

            lookup_index = pd.MultiIndex.from_frame(out[[key, "__history_date"]])
            stats_index = pd.MultiIndex.from_frame(stats[[key, "__history_date"]])
            rate_lookup = pd.Series(stats[feature].to_numpy(), index=stats_index)
            count_lookup = pd.Series(stats["prior_count"].to_numpy(), index=stats_index)
            out[feature] = rate_lookup.reindex(lookup_index).to_numpy(dtype=float)
            ordered_count_by_key[key] = pd.Series(
                count_lookup.reindex(lookup_index).to_numpy(dtype=float), index=out.index
            ).fillna(0.0)

        for key, feature, _ in SHARE_SPECS:
            prior_count = ordered_count_by_key.get(
                key, pd.Series(0.0, index=out.index, dtype=float)
            )
            denominator = row_global_count.replace(0, np.nan)
            out[feature] = (prior_count / denominator).fillna(self.frequency_fallback)

        out = out.drop(columns=["__history_date"])
        self.fit(frame)
        self.ordered_encoding = True
        return out

    def _fit_transform_leave_one_out(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback used only when a training frame has no date column."""
        out = train_df.copy()
        global_rate = float(out[TARGET_COL].mean())
        n_total = len(out)
        for key, feature, _, _ in RATE_SPECS:
            if key not in out.columns:
                out[feature] = global_rate
                continue
            sums = out.groupby(key, observed=True)[TARGET_COL].transform("sum")
            counts = out.groupby(key, observed=True)[TARGET_COL].transform("count")
            out[feature] = (
                sums - out[TARGET_COL] + self.smoothing_strength * global_rate
            ) / (counts - 1 + self.smoothing_strength)

        count_feature = {
            key: out.groupby(key, observed=True)[TARGET_COL].transform("count") - 1
            for key, _, _ in SHARE_SPECS
            if key in out.columns
        }
        for key, feature, _ in SHARE_SPECS:
            counts = count_feature.get(key, pd.Series(0, index=out.index))
            out[feature] = counts / max(n_total - 1, 1)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted smoothed maps to future rows."""
        out = _ensure_keys(df)
        for key, feature, rate_attr, count_attr in RATE_SPECS:
            out[feature] = _map_with_fallback(
                out, key, getattr(self, rate_attr), self.global_fallback
            )
            count_feature = feature.replace("DelayRate", "HistoryCount")
            out[count_feature] = _map_with_fallback(out, key, getattr(self, count_attr), 0.0)

        for key, feature, share_attr in SHARE_SPECS:
            out[feature] = _map_with_fallback(
                out, key, getattr(self, share_attr), self.frequency_fallback
            )
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
            "smoothing_strength": float(
                data.get("smoothing_strength", cls.DEFAULT_SMOOTHING_STRENGTH)
            ),
            "ordered_encoding": bool(data.get("ordered_encoding", False)),
        }
        for _, _, rate_attr, count_attr in RATE_SPECS:
            kwargs[rate_attr] = data.get(rate_attr, {})
            kwargs[count_attr] = {
                str(key): int(value) for key, value in data.get(count_attr, {}).items()
            }
        for _, _, share_attr in SHARE_SPECS:
            kwargs[share_attr] = data.get(share_attr, {})
        return cls(**kwargs)
