"""
Historical delay-rate and schedule-density aggregate features.

All aggregates are fit on the training split only, then applied to validation,
test and inference rows. v6.1 adds richer pre-flight signal: carrier-route,
airport-hour and schedule-volume features that are known before departure when
computed from historical/schedule data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import GLOBAL_FALLBACK_DELAY_RATE, TARGET_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _share_map(df: pd.DataFrame, key: str) -> dict[str, float]:
    if key not in df.columns:
        return {}
    total = max(len(df), 1)
    return (df.groupby(key).size() / total).to_dict()


def _mean_map(df: pd.DataFrame, key: str) -> dict[str, float]:
    if key not in df.columns:
        return {}
    return df.groupby(key)[TARGET_COL].mean().to_dict()


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


def _map_with_fallback(df: pd.DataFrame, key: str, mapping: dict[str, float], fallback: float) -> pd.Series:
    if key not in df.columns:
        return pd.Series(fallback, index=df.index, dtype=float)
    return df[key].map(mapping).fillna(fallback)


@dataclass
class HistoricalAggregates:
    """Learned historical delay rates and frequency shares, fit on training data only."""

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

    route_shares: dict[str, float] = field(default_factory=dict)
    carrier_route_shares: dict[str, float] = field(default_factory=dict)
    airline_origin_shares: dict[str, float] = field(default_factory=dict)
    origin_hour_shares: dict[str, float] = field(default_factory=dict)
    dest_hour_shares: dict[str, float] = field(default_factory=dict)
    carrier_dep_hour_shares: dict[str, float] = field(default_factory=dict)

    global_fallback: float = GLOBAL_FALLBACK_DELAY_RATE
    frequency_fallback: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HistoricalAggregates":
        """Compute aggregate signal from the training split only."""
        train_df = _ensure_keys(train_df)
        self.global_fallback = float(train_df[TARGET_COL].mean())

        self.carrier_rates = _mean_map(train_df, "Airline")
        self.route_rates = _mean_map(train_df, "Route")
        self.origin_rates = _mean_map(train_df, "Origin")
        self.dest_rates = _mean_map(train_df, "Dest")

        self.carrier_route_rates = _mean_map(train_df, "CarrierRoute")
        self.airline_origin_rates = _mean_map(train_df, "AirlineOrigin")
        self.airline_dest_rates = _mean_map(train_df, "AirlineDest")
        self.origin_hour_rates = _mean_map(train_df, "OriginDepHour")
        self.dest_hour_rates = _mean_map(train_df, "DestArrHour")
        self.carrier_dep_hour_rates = _mean_map(train_df, "CarrierDepHour")

        self.route_shares = _share_map(train_df, "Route")
        self.carrier_route_shares = _share_map(train_df, "CarrierRoute")
        self.airline_origin_shares = _share_map(train_df, "AirlineOrigin")
        self.origin_hour_shares = _share_map(train_df, "OriginDepHour")
        self.dest_hour_shares = _share_map(train_df, "DestArrHour")
        self.carrier_dep_hour_shares = _share_map(train_df, "CarrierDepHour")

        logger.info(
            "Fit historical aggregates v6.1: %d carriers, %d routes, %d carrier-routes, "
            "%d origin-hour keys, %d destination-hour keys, global fallback=%.4f",
            len(self.carrier_rates),
            len(self.route_rates),
            len(self.carrier_route_rates),
            len(self.origin_hour_rates),
            len(self.dest_hour_rates),
            self.global_fallback,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned rates to any dataframe (train, test, or single inference row)."""
        df = _ensure_keys(df)
        df["CarrierDelayRate"] = _map_with_fallback(df, "Airline", self.carrier_rates, self.global_fallback)
        df["RouteDelayRate"] = _map_with_fallback(df, "Route", self.route_rates, self.global_fallback)
        df["OriginDelayRate"] = _map_with_fallback(df, "Origin", self.origin_rates, self.global_fallback)
        df["DestDelayRate"] = _map_with_fallback(df, "Dest", self.dest_rates, self.global_fallback)

        df["CarrierRouteDelayRate"] = _map_with_fallback(df, "CarrierRoute", self.carrier_route_rates, self.global_fallback)
        df["AirlineOriginDelayRate"] = _map_with_fallback(df, "AirlineOrigin", self.airline_origin_rates, self.global_fallback)
        df["AirlineDestDelayRate"] = _map_with_fallback(df, "AirlineDest", self.airline_dest_rates, self.global_fallback)
        df["OriginHourDelayRate"] = _map_with_fallback(df, "OriginDepHour", self.origin_hour_rates, self.global_fallback)
        df["DestHourDelayRate"] = _map_with_fallback(df, "DestArrHour", self.dest_hour_rates, self.global_fallback)
        df["CarrierDepHourDelayRate"] = _map_with_fallback(df, "CarrierDepHour", self.carrier_dep_hour_rates, self.global_fallback)

        df["RouteFlightShare"] = _map_with_fallback(df, "Route", self.route_shares, self.frequency_fallback)
        df["CarrierRouteFlightShare"] = _map_with_fallback(df, "CarrierRoute", self.carrier_route_shares, self.frequency_fallback)
        df["AirlineOriginFlightShare"] = _map_with_fallback(df, "AirlineOrigin", self.airline_origin_shares, self.frequency_fallback)
        df["OriginHourFlightShare"] = _map_with_fallback(df, "OriginDepHour", self.origin_hour_shares, self.frequency_fallback)
        df["DestHourFlightShare"] = _map_with_fallback(df, "DestArrHour", self.dest_hour_shares, self.frequency_fallback)
        df["CarrierDepHourFlightShare"] = _map_with_fallback(df, "CarrierDepHour", self.carrier_dep_hour_shares, self.frequency_fallback)
        return df

    def lookup_single(self, airline: str, origin: str, dest: str, route: str) -> dict[str, float]:
        """Convenience lookup for a single inference request (backwards compatible subset)."""
        return {
            "CarrierDelayRate": self.carrier_rates.get(airline, self.global_fallback),
            "RouteDelayRate": self.route_rates.get(route, self.global_fallback),
            "OriginDelayRate": self.origin_rates.get(origin, self.global_fallback),
            "DestDelayRate": self.dest_rates.get(dest, self.global_fallback),
        }

    def to_dict(self) -> dict:
        return {
            "carrier_rates": self.carrier_rates,
            "route_rates": self.route_rates,
            "origin_rates": self.origin_rates,
            "dest_rates": self.dest_rates,
            "carrier_route_rates": self.carrier_route_rates,
            "airline_origin_rates": self.airline_origin_rates,
            "airline_dest_rates": self.airline_dest_rates,
            "origin_hour_rates": self.origin_hour_rates,
            "dest_hour_rates": self.dest_hour_rates,
            "carrier_dep_hour_rates": self.carrier_dep_hour_rates,
            "route_shares": self.route_shares,
            "carrier_route_shares": self.carrier_route_shares,
            "airline_origin_shares": self.airline_origin_shares,
            "origin_hour_shares": self.origin_hour_shares,
            "dest_hour_shares": self.dest_hour_shares,
            "carrier_dep_hour_shares": self.carrier_dep_hour_shares,
            "global_fallback": self.global_fallback,
            "frequency_fallback": self.frequency_fallback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoricalAggregates":
        return cls(
            carrier_rates=data.get("carrier_rates", {}),
            route_rates=data.get("route_rates", {}),
            origin_rates=data.get("origin_rates", {}),
            dest_rates=data.get("dest_rates", {}),
            carrier_route_rates=data.get("carrier_route_rates", {}),
            airline_origin_rates=data.get("airline_origin_rates", {}),
            airline_dest_rates=data.get("airline_dest_rates", {}),
            origin_hour_rates=data.get("origin_hour_rates", {}),
            dest_hour_rates=data.get("dest_hour_rates", {}),
            carrier_dep_hour_rates=data.get("carrier_dep_hour_rates", {}),
            route_shares=data.get("route_shares", {}),
            carrier_route_shares=data.get("carrier_route_shares", {}),
            airline_origin_shares=data.get("airline_origin_shares", {}),
            origin_hour_shares=data.get("origin_hour_shares", {}),
            dest_hour_shares=data.get("dest_hour_shares", {}),
            carrier_dep_hour_shares=data.get("carrier_dep_hour_shares", {}),
            global_fallback=data.get("global_fallback", GLOBAL_FALLBACK_DELAY_RATE),
            frequency_fallback=data.get("frequency_fallback", 0.0),
        )
