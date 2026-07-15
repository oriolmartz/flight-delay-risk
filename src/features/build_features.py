"""Leakage-safe features available from a published flight schedule."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config import FORBIDDEN_LEAKAGE_COLUMNS


def hhmm_to_hour(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 2400)
    return ((numeric // 100).astype(int) % 24).astype(int)


def hhmm_to_minute(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 2400).astype(int)
    hour = (numeric // 100) % 24
    minute = (numeric % 100).clip(0, 59)
    return (hour * 60 + minute).astype(int)


def build_route(origin: pd.Series, dest: pd.Series) -> pd.Series:
    return origin.astype(str).str.upper() + "_" + dest.astype(str).str.upper()


def _distance_band(distance: pd.Series) -> pd.Series:
    values = pd.to_numeric(distance, errors="coerce").fillna(0)
    return pd.cut(
        values,
        bins=[-1, 250, 500, 1000, 2000, 4000, float("inf")],
        labels=["very_short", "short", "medium", "long", "very_long", "ultra_long"],
    ).astype(str)


def _period_from_hour(hour: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour, errors="coerce").fillna(0).astype(int) % 24
    result = pd.Series("midday", index=h.index, dtype="object")
    result[(h >= 5) & (h <= 8)] = "early_morning"
    result[(h >= 9) & (h <= 11)] = "morning"
    result[(h >= 12) & (h <= 15)] = "afternoon"
    result[(h >= 16) & (h <= 19)] = "evening_peak"
    result[(h >= 20) & (h <= 23)] = "night"
    result[(h >= 0) & (h <= 4)] = "red_eye"
    return result


def _calendar_dates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "FlightDate" in df.columns:
        dates = pd.to_datetime(df["FlightDate"], errors="coerce", format="mixed").dt.normalize()
        known = dates.notna().astype(int)
    else:
        dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        known = pd.Series(0, index=df.index, dtype=int)
    year_source = df["Year"] if "Year" in df.columns else pd.Series(2024, index=df.index)
    month_source = df["Month"] if "Month" in df.columns else pd.Series(1, index=df.index)
    year = pd.to_numeric(year_source, errors="coerce").fillna(2024).astype(int)
    month = pd.to_numeric(month_source, errors="coerce").fillna(1).clip(1, 12).astype(int)
    fallback = pd.to_datetime(
        {"year": year, "month": month, "day": pd.Series(15, index=df.index)}, errors="coerce"
    )
    return dates.fillna(fallback), known


def _holiday_distance(dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    if dates.empty:
        empty = pd.Series(dtype=float, index=dates.index)
        return empty, empty.astype(int)
    start = dates.min() - pd.Timedelta(days=370)
    end = dates.max() + pd.Timedelta(days=370)
    holidays = USFederalHolidayCalendar().holidays(start=start, end=end).normalize()
    holiday_values = holidays.to_numpy(dtype="datetime64[D]")
    day_values = dates.to_numpy(dtype="datetime64[D]")
    distances = np.array(
        [min(abs((day - holiday_values).astype("timedelta64[D]").astype(int))) for day in day_values],
        dtype=int,
    )
    return pd.Series(distances, index=dates.index), dates.isin(holidays).astype(int)


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Airline" not in df.columns:
        raise KeyError("Expected an 'Airline' column (Reporting/Operating carrier code).")
    for column in ("Airline", "Origin", "Dest"):
        df[column] = df[column].astype(str).str.upper().str.strip()

    df["DepHour"] = hhmm_to_hour(df["CRSDepTime"])
    df["ArrHour"] = hhmm_to_hour(df["CRSArrTime"])
    df["DepMinute"] = hhmm_to_minute(df["CRSDepTime"])
    df["ArrMinute"] = hhmm_to_minute(df["CRSArrTime"])
    df["Route"] = build_route(df["Origin"], df["Dest"])
    df["IsWeekend"] = pd.to_numeric(df["DayOfWeek"], errors="coerce").isin([6, 7]).astype(int)
    df["DepPeriod"] = _period_from_hour(df["DepHour"])
    df["ArrPeriod"] = _period_from_hour(df["ArrHour"])
    df["DistanceBand"] = _distance_band(df["Distance"])
    df["IsMorningPeak"] = df["DepHour"].between(6, 9).astype(int)
    df["IsEveningPeak"] = df["DepHour"].between(16, 19).astype(int)
    df["IsPeakHour"] = ((df["IsMorningPeak"] == 1) | (df["IsEveningPeak"] == 1)).astype(int)
    df["IsRedEye"] = ((df["DepHour"] <= 5) | (df["DepHour"] >= 22)).astype(int)

    elapsed = pd.to_numeric(df["CRSElapsedTime"], errors="coerce").replace(0, np.nan)
    distance = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    df["IsLongHaul"] = (distance >= 2000).astype(int)
    df["ScheduledSpeedMph"] = (distance / (elapsed / 60)).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["LogDistance"] = np.log1p(distance.clip(lower=0))
    for name, values, period in (
        ("DepHour", df["DepHour"], 24), ("ArrHour", df["ArrHour"], 24),
        ("DepMinute", df["DepMinute"], 1440), ("ArrMinute", df["ArrMinute"], 1440),
    ):
        df[f"{name}Sin"] = np.sin(2 * np.pi * values / period)
        df[f"{name}Cos"] = np.cos(2 * np.pi * values / period)
    df["IsOvernightSchedule"] = (df["ArrMinute"] < df["DepMinute"]).astype(int)

    dates, date_known = _calendar_dates(df)
    df["CalendarDateKnown"] = date_known
    df["DayOfMonth"] = dates.dt.day.astype(int)
    df["DayOfYear"] = dates.dt.dayofyear.astype(int)
    df["WeekOfYear"] = dates.dt.isocalendar().week.astype(int)
    df["Quarter"] = dates.dt.quarter.astype(int)
    df["YearProgress"] = (df["DayOfYear"] - 1) / np.where(dates.dt.is_leap_year, 365, 364)
    month = pd.to_numeric(df["Month"], errors="coerce").fillna(dates.dt.month).astype(int)
    dow = pd.to_numeric(df["DayOfWeek"], errors="coerce").fillna(dates.dt.dayofweek + 1).astype(int)
    df["MonthSin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["MonthCos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["DayOfWeekSin"] = np.sin(2 * np.pi * (dow - 1) / 7)
    df["DayOfWeekCos"] = np.cos(2 * np.pi * (dow - 1) / 7)
    df["YearDaySin"] = np.sin(2 * np.pi * (df["DayOfYear"] - 1) / 366)
    df["YearDayCos"] = np.cos(2 * np.pi * (df["DayOfYear"] - 1) / 366)
    df["Season"] = month.map({12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}).fillna("unknown")
    holiday_distance, is_holiday = _holiday_distance(dates)
    df["DaysToNearestFederalHoliday"] = holiday_distance.astype(float)
    df["IsFederalHoliday"] = is_holiday.astype(int)
    df["IsHolidayWindow"] = (holiday_distance <= 2).astype(int)

    df["CarrierRoute"] = df["Airline"] + "_" + df["Route"]
    df["AirlineOrigin"] = df["Airline"] + "_" + df["Origin"]
    df["AirlineDest"] = df["Airline"] + "_" + df["Dest"]
    df["OriginDepHour"] = df["Origin"] + "_" + df["DepHour"].astype(str)
    df["DestArrHour"] = df["Dest"] + "_" + df["ArrHour"].astype(str)
    df["CarrierDepHour"] = df["Airline"] + "_" + df["DepHour"].astype(str)
    return df


def assert_no_leakage_columns(feature_columns: list[str]) -> None:
    leaked = set(feature_columns) & set(FORBIDDEN_LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage detected in feature set: {sorted(leaked)}")
