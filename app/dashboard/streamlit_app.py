"""FlightRisk v1.0 public product dashboard.

Four product surfaces:
1. analyze one scheduled flight,
2. validate and rank a schedule,
3. inspect temporal validation evidence,
4. inspect model lineage, monitoring and release performance.
"""
from __future__ import annotations

import html
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.dashboard.i18n import TEXT
from app.dashboard.theme import CSS
from app.services import prediction_service, report_service
from src.models.predict import PredictionInput
from src.version import APP_VERSION, RELEASE_NAME

st.set_page_config(
    page_title="FlightRisk - Pre-departure Risk Workbench",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_schedule.csv"
TEMPLATE_PATH = ROOT / "data" / "sample" / "schedule_template.csv"
MAX_UPLOAD_ROWS = 500


@dataclass
class ScheduleValidationResult:
    prepared: pd.DataFrame
    payloads: list[PredictionInput]
    errors: pd.DataFrame


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _language_selector() -> tuple[str, dict[str, Any]]:
    choice = st.selectbox(
        "Language / Idioma",
        ["English", "Español"],
        index=0,
        key="language_selector",
        label_visibility="collapsed",
    )
    lang = "en" if choice == "English" else "es"
    return lang, TEXT[lang]


def _safe_model_info() -> dict[str, Any]:
    try:
        return prediction_service.model_info()
    except Exception:
        return {}


def _safe_model_card() -> dict[str, Any]:
    try:
        return prediction_service.model_card()
    except Exception:
        return {}


def _safe_catalog() -> dict[str, list[str]]:
    fallback = {
        "carriers": ["AA", "AS", "B6", "DL", "F9", "G4", "HA", "NK", "UA", "WN"],
        "airports": ["ATL", "BOS", "CLT", "DEN", "DFW", "EWR", "JFK", "LAX", "MIA", "ORD", "SFO"],
    }
    try:
        catalog = prediction_service.input_catalog()
        return {
            "carriers": catalog.get("carriers") or fallback["carriers"],
            "airports": catalog.get("airports") or fallback["airports"],
        }
    except Exception:
        return fallback


def _sample_input() -> PredictionInput:
    return PredictionInput(
        airline="DL",
        origin="JFK",
        destination="LAX",
        month=7,
        day_of_week=5,
        crs_dep_time=1830,
        crs_arr_time=2145,
        crs_elapsed_time=375,
        distance=2475,
    )


def _safe_sample_state(model_available: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    fallback_prediction = {
        "delay_probability": 0.168,
        "raw_model_score": 0.501,
        "calibration_method": "isotonic",
        "risk_level": "low",
        "decision_threshold": 0.17,
        "top_factors": ["sample schedule profile"],
        "local_contributions": [],
    }
    fallback_context = {
        "route": "JFK → LAX",
        "global_rate": 0.227,
        "route_rate": 0.162,
        "route_support": 3842,
        "route_support_estimate": 3842,
        "route_seen": True,
        "signals": [],
    }
    if not model_available:
        return fallback_prediction, fallback_context
    try:
        payload = _sample_input()
        return prediction_service.predict_flight(payload), prediction_service.prediction_context(payload)
    except Exception:
        return fallback_prediction, fallback_context


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_int(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{int(round(value)):,}"


def _fmt_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.1f} ms"


def _hhmm(value: time) -> int:
    return value.hour * 100 + value.minute


def _clock_label(value: Any) -> str:
    try:
        numeric = int(float(value))
        if numeric == 2400:
            return "00:00"
        hour, minute = divmod(numeric, 100)
        return f"{hour:02d}:{minute:02d}"
    except (TypeError, ValueError):
        return str(value)


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="fr-section-head"><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def _decision_copy(level: str, t: dict[str, Any]) -> tuple[str, str]:
    block = t["decision"].get(level, t["decision"]["moderate"])
    return str(block[0]), str(block[1])


# ---------------------------------------------------------------------------
# Product shell
# ---------------------------------------------------------------------------


def _topbar(model_available: bool, artifact_version: str | None, t: dict[str, Any]) -> None:
    status_class = "ok" if model_available else "warn"
    status_text = t["topbar"]["model_loaded"] if model_available else t["topbar"]["model_unavailable"]
    artifact = artifact_version or "unknown"
    st.markdown(
        f"""
<div class="fr-topbar">
  <div class="fr-brand">
    <div class="fr-mark">FR</div>
    <div>
      <div class="fr-brand-name">FLIGHTRISK</div>
      <div class="fr-byline">{html.escape(str(t['topbar']['byline']))}</div>
    </div>
  </div>
  <div class="fr-status">
    <span class="fr-chip release">v{APP_VERSION} · {html.escape(str(t['topbar']['public_release']))}</span>
    <span class="fr-chip">artifact {html.escape(str(artifact))}</span>
    <span class="fr-chip {status_class}">{html.escape(str(status_text))}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _workflow(t: dict[str, Any]) -> None:
    steps = t["workflow"]
    blocks = "".join(
        f'<div class="fr-step"><span>{idx}</span><b>{html.escape(str(label))}</b></div>'
        for idx, label in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="fr-workflow">{blocks}</div>', unsafe_allow_html=True)


def _hero(t: dict[str, Any], model_available: bool) -> None:
    prediction, context = _safe_sample_state(model_available)
    probability = float(prediction.get("delay_probability", 0.0))
    route_rate = float(context.get("route_rate", 0.0))
    global_rate = float(context.get("global_rate", 0.0))
    relative = probability / route_rate if route_rate > 0 else probability / max(global_rate, 1e-9)
    support = context.get("route_support", context.get("route_support_estimate"))
    route = context.get("route", "JFK → LAX")
    level = str(prediction.get("risk_level", "moderate"))
    priority = {
        "low": t["hero_card"]["routine"],
        "moderate": t["hero_card"]["watch"],
        "high": t["hero_card"]["priority"],
    }.get(level, t["hero_card"]["watch"])
    calibration_method = str(prediction.get("calibration_method", "identity"))

    st.markdown('<div class="fr-hero">', unsafe_allow_html=True)
    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        st.markdown(
            f"""
<div class="fr-kicker">{t['hero_kicker']}</div>
<div class="fr-title">{t['hero_title']}</div>
<div class="fr-subtitle">{t['hero_sub']}</div>
<div class="fr-constraint">{t['constraint']}</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
<div class="fr-flight-card">
  <div class="fr-flight-top">
    <div>
      <div class="fr-flight-id">{t['hero_card']['example']}</div>
      <div class="fr-route">{html.escape(str(route))}</div>
    </div>
    <div class="fr-priority">{html.escape(str(priority))}</div>
  </div>
  <div class="fr-risk-number">{_fmt_pct(probability)}</div>
  <div class="fr-risk-label">{t['hero_card']['probability']} · {calibration_method}</div>
  <div class="fr-flight-grid">
    <div class="fr-flight-stat"><b>{_fmt_pct(route_rate)}</b><span>{t['hero_card']['route_rate']}</span></div>
    <div class="fr-flight-stat"><b>{relative:.2f}×</b><span>{t['hero_card']['relative']}</span></div>
    <div class="fr-flight-stat"><b>{_fmt_int(support)}</b><span>{t['hero_card']['support']}</span></div>
    <div class="fr-flight-stat"><b>{_fmt_pct(global_rate)}</b><span>{t['hero_card']['fallback']}</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    _workflow(t)


# ---------------------------------------------------------------------------
# Single-flight workflow
# ---------------------------------------------------------------------------


def _flight_form(
    lang: str,
    t: dict[str, Any],
    catalog: dict[str, list[str]],
    disabled: bool,
) -> tuple[PredictionInput, dict[str, Any]] | None:
    carriers = catalog["carriers"]
    airports = catalog["airports"]
    default_carrier = carriers.index("DL") if "DL" in carriers else 0
    default_origin = airports.index("JFK") if "JFK" in airports else 0
    default_dest = airports.index("LAX") if "LAX" in airports else min(1, len(airports) - 1)
    labels = t["form"]

    with st.form("single_flight_form"):
        c1, c2, c3, c4 = st.columns([0.18, 0.18, 0.32, 0.32])
        airline = c1.selectbox(labels["carrier"], carriers, index=default_carrier)
        flight_number = c2.text_input(labels["flight_number"], value="418", max_chars=8)
        origin = c3.selectbox(labels["origin"], airports, index=default_origin)
        destination = c4.selectbox(labels["destination"], airports, index=default_dest)

        c5, c6, c7 = st.columns(3)
        flight_date = c5.date_input(labels["date"], value=date.today())
        departure = c6.time_input(labels["departure"], value=time(18, 30), step=300)
        arrival = c7.time_input(labels["arrival"], value=time(21, 45), step=300)

        c8, c9 = st.columns(2)
        duration = c8.number_input(labels["duration"], min_value=20, max_value=900, value=375, step=5)
        distance = c9.number_input(labels["distance"], min_value=20.0, max_value=10000.0, value=2475.0, step=25.0)
        submitted = st.form_submit_button(labels["submit"], disabled=disabled, width="stretch")

    if not submitted:
        return None
    if origin == destination:
        st.error(labels["route_error"])
        return None

    payload = PredictionInput(
        airline=airline,
        origin=origin,
        destination=destination,
        month=flight_date.month,
        day_of_week=flight_date.isoweekday(),
        crs_dep_time=_hhmm(departure),
        crs_arr_time=_hhmm(arrival),
        crs_elapsed_time=int(duration),
        distance=float(distance),
    )
    metadata = {
        "airline": airline,
        "flight_number": flight_number.strip(),
        "origin": origin,
        "destination": destination,
        "flight_date": flight_date.isoformat(),
        "scheduled_departure": departure.strftime("%H:%M"),
        "scheduled_arrival": arrival.strftime("%H:%M"),
        "scheduled_duration_minutes": int(duration),
        "distance_miles": float(distance),
        "lang": lang,
    }
    return payload, metadata


def _render_context_rows(context: dict[str, Any], t: dict[str, Any]) -> None:
    signals = context.get("signals") or []
    for signal in signals[:4]:
        value = signal.get("value")
        baseline = signal.get("baseline")
        support = signal.get("support")
        direction = t["single"]["above"] if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and value >= baseline else t["single"]["below"]
        support_copy = f" · {_fmt_int(support)} {t['single']['rows']}" if support else ""
        st.markdown(
            f"""
<div class="fr-context-row">
  <div>
    <div class="fr-context-label">{html.escape(str(signal.get('label', 'Historical context')))}</div>
    <div class="fr-context-meta">{html.escape(str(direction))}{html.escape(support_copy)}</div>
  </div>
  <div class="fr-context-value">{_fmt_pct(value)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _contribution_label(item: dict[str, Any], t: dict[str, Any]) -> tuple[str, str]:
    feature = str(item.get("feature", "feature"))
    label = str(t["features"].get(feature, feature))
    raw = item.get("active_category")
    if raw is None:
        raw = item.get("raw_value")
    raw_label = "" if raw is None else str(raw)
    return label, raw_label


def _render_contributions(result: dict[str, Any], t: dict[str, Any]) -> None:
    contributions = result.get("local_contributions") or []
    st.markdown(f"### {t['single']['explanation']}")
    st.markdown(f'<div class="fr-note">{t["single"]["explanation_note"]}</div>', unsafe_allow_html=True)
    if not contributions:
        return

    for item in contributions[:6]:
        contribution = float(item.get("contribution", 0.0))
        direction_class = "up" if contribution >= 0 else "down"
        direction_label = t["single"]["increase"] if contribution >= 0 else t["single"]["decrease"]
        label, raw_label = _contribution_label(item, t)
        st.markdown(
            f"""
<div class="fr-contribution">
  <div><b>{html.escape(label)}</b><span>{html.escape(raw_label)}</span></div>
  <div class="fr-contribution-value {direction_class}">{html.escape(str(direction_label))} · {contribution:+.3f}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_prediction(payload: PredictionInput, metadata: dict[str, Any], lang: str, t: dict[str, Any]) -> None:
    try:
        result = prediction_service.predict_flight(payload)
        context = prediction_service.prediction_context(payload)
    except Exception as exc:
        st.error(f"{t['single']['prediction_failed']}: {exc}")
        return

    probability = float(result["delay_probability"])
    raw_score = float(result.get("raw_model_score", probability))
    calibration_method = str(result.get("calibration_method", "identity"))
    route_rate = float(context.get("route_rate", context.get("global_rate", 0.0)))
    global_rate = float(context.get("global_rate", 0.0))
    relative = probability / route_rate if route_rate > 0 else 0.0
    decision, explanation = _decision_copy(str(result.get("risk_level", "moderate")), t)
    metadata["review_label"] = decision

    st.markdown(
        f"""
<div class="fr-decision">
  <div class="fr-decision-kicker">{t['decision']['kicker']}</div>
  <h3>{html.escape(decision)}</h3>
  <p>{html.escape(explanation)}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["metrics"]["probability"], _fmt_pct(probability))
    m2.metric(t["metrics"]["route_cohort"], _fmt_pct(route_rate))
    m3.metric(t["metrics"]["relative"], f"{relative:.2f}×")
    m4.metric(t["metrics"]["support"], _fmt_int(context.get("route_support")))

    left, right = st.columns([0.57, 0.43], gap="large")
    with left:
        st.markdown(f'<div class="fr-panel"><b>{t["single"]["historical_context"]}</b>', unsafe_allow_html=True)
        _render_context_rows(context, t)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        route_seen = t["single"]["seen"] if context.get("route_seen") else t["single"]["unseen"]
        carrier_route_seen = t["single"]["seen"] if context.get("carrier_route_seen") else t["single"]["unseen"]
        st.markdown(
            f"""
<div class="fr-panel">
  <b>{t['single']['reliability']}</b>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['coverage']}</div><div class="fr-context-value">{route_seen}</div></div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['carrier_route_coverage']}</div><div class="fr-context-value">{carrier_route_seen}</div></div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['calibration']}</div><div class="fr-context-value">{calibration_method}</div></div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['raw_score']}</div><div class="fr-context-value">{_fmt_pct(raw_score)}</div></div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['fallback']}</div><div class="fr-context-value">{_fmt_pct(global_rate)}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    _render_contributions(result, t)

    pdf = report_service.build_flight_brief_pdf(metadata, result, context, lang=lang)
    filename = f"flightrisk_{metadata['airline']}_{metadata.get('flight_number') or 'flight'}_{lang}.pdf"
    st.download_button(
        t["single"]["download_pdf"],
        data=pdf,
        file_name=filename,
        mime="application/pdf",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Schedule upload, validation and ranking
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in normalized.columns
    ]
    aliases = {
        "reporting_airline": "airline",
        "operating_airline": "airline",
        "carrier": "airline",
        "dest": "destination",
        "origin_airport": "origin",
        "destination_airport": "destination",
        "crsdeptime": "crs_dep_time",
        "crsarrtime": "crs_arr_time",
        "crselapsedtime": "crs_elapsed_time",
        "dayofweek": "day_of_week",
        "date": "flight_date",
        "departure": "scheduled_departure",
        "arrival": "scheduled_arrival",
        "duration_minutes": "scheduled_duration_minutes",
        "duration": "scheduled_duration_minutes",
        "distance": "distance_miles",
        "flight": "flight_number",
    }
    return normalized.rename(columns=aliases)


def _parse_clock(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("missing scheduled time")
    if isinstance(value, time):
        return _hhmm(value)
    if isinstance(value, pd.Timestamp):
        return value.hour * 100 + value.minute
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) < 2:
            raise ValueError("invalid time")
        hour = int(parts[0])
        minute = int(parts[1][:2])
        value_int = hour * 100 + minute
    else:
        value_int = int(float(text))
    if value_int == 2400:
        return value_int
    hour, minute = divmod(value_int, 100)
    if hour > 23 or minute > 59 or value_int < 0:
        raise ValueError("time must use HH:MM or HHMM")
    return value_int


def _row_error(row_number: int, reason: str) -> dict[str, Any]:
    return {"row": row_number, "reason": reason}


def validate_schedule_dataframe(df: pd.DataFrame) -> ScheduleValidationResult:
    normalized = _normalize_columns(df)
    if len(normalized) > MAX_UPLOAD_ROWS:
        return ScheduleValidationResult(
            pd.DataFrame(),
            [],
            pd.DataFrame([_row_error(0, f"maximum {MAX_UPLOAD_ROWS} rows per upload")]),
        )

    natural_required = {
        "airline",
        "origin",
        "destination",
        "flight_date",
        "scheduled_departure",
        "scheduled_arrival",
        "scheduled_duration_minutes",
        "distance_miles",
    }
    legacy_required = {
        "airline",
        "origin",
        "destination",
        "month",
        "day_of_week",
        "crs_dep_time",
        "crs_arr_time",
        "crs_elapsed_time",
        "distance",
    }

    is_natural = natural_required.issubset(normalized.columns)
    is_legacy = legacy_required.issubset(normalized.columns)
    if not is_natural and not is_legacy:
        missing_natural = sorted(natural_required - set(normalized.columns))
        return ScheduleValidationResult(
            pd.DataFrame(),
            [],
            pd.DataFrame([_row_error(0, "missing columns: " + ", ".join(missing_natural))]),
        )

    valid_rows: list[dict[str, Any]] = []
    payloads: list[PredictionInput] = []
    errors: list[dict[str, Any]] = []

    for idx, row in normalized.iterrows():
        source_row = int(idx) + 2
        try:
            airline = str(row["airline"]).strip().upper()
            origin = str(row["origin"]).strip().upper()
            destination = str(row["destination"]).strip().upper()
            if not airline or airline == "NAN":
                raise ValueError("missing airline")
            if len(origin) != 3 or len(destination) != 3:
                raise ValueError("origin and destination must be 3-letter IATA codes")
            if origin == destination:
                raise ValueError("origin and destination must be different")

            if is_natural:
                flight_date = pd.to_datetime(row["flight_date"], errors="raise")
                dep_hhmm = _parse_clock(row["scheduled_departure"])
                arr_hhmm = _parse_clock(row["scheduled_arrival"])
                duration = int(float(row["scheduled_duration_minutes"]))
                distance = float(row["distance_miles"])
                month = int(flight_date.month)
                day_of_week = int(flight_date.isoweekday())
                date_label = flight_date.date().isoformat()
                dep_label = _clock_label(dep_hhmm)
                arr_label = _clock_label(arr_hhmm)
            else:
                month = int(float(row["month"]))
                day_of_week = int(float(row["day_of_week"]))
                dep_hhmm = _parse_clock(row["crs_dep_time"])
                arr_hhmm = _parse_clock(row["crs_arr_time"])
                duration = int(float(row["crs_elapsed_time"]))
                distance = float(row["distance"])
                date_label = ""
                dep_label = _clock_label(dep_hhmm)
                arr_label = _clock_label(arr_hhmm)

            if not 1 <= month <= 12:
                raise ValueError("month must be between 1 and 12")
            if not 1 <= day_of_week <= 7:
                raise ValueError("day_of_week must be between 1 and 7")
            if not 20 <= duration <= 900:
                raise ValueError("scheduled duration must be between 20 and 900 minutes")
            if not 20 <= distance <= 10000:
                raise ValueError("distance must be between 20 and 10,000 miles")

            payloads.append(
                PredictionInput(
                    airline=airline,
                    origin=origin,
                    destination=destination,
                    month=month,
                    day_of_week=day_of_week,
                    crs_dep_time=dep_hhmm,
                    crs_arr_time=arr_hhmm,
                    crs_elapsed_time=duration,
                    distance=distance,
                )
            )
            valid_rows.append(
                {
                    "source_row": source_row,
                    "flight_number": str(row.get("flight_number", "")).strip().replace("nan", ""),
                    "airline": airline,
                    "origin": origin,
                    "destination": destination,
                    "flight_date": date_label,
                    "scheduled_departure": dep_label,
                    "scheduled_arrival": arr_label,
                    "scheduled_duration_minutes": duration,
                    "distance_miles": distance,
                    "month": month,
                    "day_of_week": day_of_week,
                    "crs_dep_time": dep_hhmm,
                    "crs_arr_time": arr_hhmm,
                    "crs_elapsed_time": duration,
                    "distance": distance,
                }
            )
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(_row_error(source_row, str(exc)))

    return ScheduleValidationResult(pd.DataFrame(valid_rows), payloads, pd.DataFrame(errors))


def _payloads_from_df(df: pd.DataFrame) -> list[PredictionInput]:
    result = validate_schedule_dataframe(df)
    if not result.errors.empty:
        raise ValueError("; ".join(result.errors["reason"].astype(str).tolist()))
    return result.payloads


def _sample_batch() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    return pd.DataFrame(
        [
            ["418", "DL", "JFK", "LAX", "2026-07-18", "18:30", "21:45", 375, 2475],
            ["218", "UA", "SFO", "EWR", "2026-07-18", "22:15", "06:30", 315, 2565],
            ["904", "AA", "ORD", "DFW", "2026-07-18", "17:20", "19:55", 155, 802],
            ["126", "WN", "ATL", "BOS", "2026-07-18", "08:05", "10:35", 150, 946],
        ],
        columns=[
            "flight_number",
            "airline",
            "origin",
            "destination",
            "flight_date",
            "scheduled_departure",
            "scheduled_arrival",
            "scheduled_duration_minutes",
            "distance_miles",
        ],
    )


def _template_batch() -> pd.DataFrame:
    if TEMPLATE_PATH.exists():
        return pd.read_csv(TEMPLATE_PATH)
    return _sample_batch().head(2)


def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validated = validate_schedule_dataframe(df)
    if not validated.errors.empty:
        raise ValueError("; ".join(validated.errors["reason"].astype(str).tolist()))
    payloads = validated.payloads
    predictions = pd.DataFrame(prediction_service.predict_flights_batch(payloads))
    contexts = pd.DataFrame(prediction_service.prediction_contexts(payloads))
    ranked = pd.concat([validated.prepared.reset_index(drop=True), predictions, contexts], axis=1)
    ranked["relative_exposure"] = ranked["delay_probability"] / ranked["route_rate"].replace(0, pd.NA)
    ranked["relative_exposure"] = ranked["relative_exposure"].fillna(0.0).astype(float)
    ranked = ranked.sort_values("delay_probability", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["schedule_percentile"] = ((len(ranked) - ranked["rank"] + 1) / max(len(ranked), 1) * 100).round(1)
    priority_count = max(1, math.ceil(len(ranked) * 0.10))
    watch_cutoff = max(priority_count, math.ceil(len(ranked) * 0.30))
    ranked["priority_tier"] = "Routine"
    ranked.loc[ranked["rank"] <= watch_cutoff, "priority_tier"] = "Watch"
    ranked.loc[ranked["rank"] <= priority_count, "priority_tier"] = "Priority"
    ranked["low_support"] = ranked["route_support"].fillna(0).astype(int) < 100
    return ranked


def _localized_queue(value: str, lang: str) -> str:
    if lang == "es":
        return {"Priority": "Prioridad", "Watch": "Vigilar", "Routine": "Rutina"}.get(value, value)
    return value


def _render_batch(model_available: bool, lang: str, t: dict[str, Any]) -> None:
    st.markdown(f'<div class="fr-note">{t["batch"]["intro"]}<br>{t["batch"]["limit"]}</div>', unsafe_allow_html=True)

    template_csv = _template_batch().to_csv(index=False)
    a, b, c = st.columns([0.54, 0.23, 0.23], gap="small")
    with a:
        uploaded = st.file_uploader(t["batch"]["upload"], type=["csv"], disabled=not model_available)
    with b:
        st.write("")
        if st.button(t["batch"]["sample"], disabled=not model_available, width="stretch"):
            st.session_state["schedule_input"] = _sample_batch()
    with c:
        st.write("")
        st.download_button(
            t["batch"]["template"],
            data=template_csv,
            file_name="flightrisk_schedule_template.csv",
            mime="text/csv",
            width="stretch",
        )

    if uploaded is not None:
        try:
            st.session_state["schedule_input"] = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(str(exc))
            return

    dataframe = st.session_state.get("schedule_input")
    if dataframe is None or not model_available:
        return

    st.markdown(f"### {t['batch']['preview']}")
    st.dataframe(pd.DataFrame(dataframe).head(12), width="stretch", hide_index=True)
    validated = validate_schedule_dataframe(pd.DataFrame(dataframe))

    v1, v2, v3, v4 = st.columns(4)
    v1.metric(t["batch"]["valid"], len(validated.prepared))
    v2.metric(t["batch"]["invalid"], len(validated.errors))
    unseen_pre = 0
    low_support_pre = 0
    if validated.payloads:
        contexts = prediction_service.prediction_contexts(validated.payloads)
        unseen_pre = sum(not bool(item.get("route_seen")) for item in contexts)
        low_support_pre = sum(int(item.get("route_support", 0)) < 100 for item in contexts)
    v3.metric(t["batch"]["unseen"], unseen_pre)
    v4.metric(t["batch"]["low_support"], low_support_pre)

    if not validated.errors.empty:
        st.warning(t["batch"]["validation_errors"])
        error_df = validated.errors.rename(columns={"row": t["batch"]["error_row"], "reason": t["batch"]["error_reason"]})
        st.dataframe(error_df, width="stretch", hide_index=True)
    else:
        st.success(t["batch"]["validation_ok"])

    if not validated.payloads:
        st.error(t["batch"]["no_valid"])
        return

    # Rank valid rows even if malformed rows were excluded.
    predictions = pd.DataFrame(prediction_service.predict_flights_batch(validated.payloads))
    contexts = pd.DataFrame(prediction_service.prediction_contexts(validated.payloads))
    ranked = pd.concat([validated.prepared.reset_index(drop=True), predictions, contexts], axis=1)
    ranked["relative_exposure"] = ranked["delay_probability"] / ranked["route_rate"].replace(0, pd.NA)
    ranked["relative_exposure"] = ranked["relative_exposure"].fillna(0.0).astype(float)
    ranked = ranked.sort_values("delay_probability", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["schedule_percentile"] = ((len(ranked) - ranked["rank"] + 1) / len(ranked) * 100).round(1)
    priority_count = max(1, math.ceil(len(ranked) * 0.10))
    watch_cutoff = max(priority_count, math.ceil(len(ranked) * 0.30))
    ranked["priority_tier"] = "Routine"
    ranked.loc[ranked["rank"] <= watch_cutoff, "priority_tier"] = "Watch"
    ranked.loc[ranked["rank"] <= priority_count, "priority_tier"] = "Priority"
    ranked["low_support"] = ranked["route_support"].fillna(0).astype(int) < 100

    total = len(ranked)
    priority = int((ranked["priority_tier"] == "Priority").sum())
    watch = int((ranked["priority_tier"] == "Watch").sum())
    avg = float(ranked["delay_probability"].mean()) if total else 0.0
    max_probability = float(ranked["delay_probability"].max()) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["batch"]["valid"], total)
    m2.metric(t["batch"]["priority"], priority)
    m3.metric(t["batch"]["watch"], watch)
    m4.metric(t["batch"]["highest"], _fmt_pct(max_probability))

    st.markdown(f"### {t['batch']['ranked']}")
    display = ranked.copy()
    display["flight"] = (display["airline"].astype(str) + " " + display["flight_number"].astype(str)).str.strip()
    display["route"] = display["origin"].astype(str) + " → " + display["destination"].astype(str)
    display["delay_probability_pct"] = (display["delay_probability"] * 100).round(1)
    display["route_rate_pct"] = (display["route_rate"] * 100).round(1)
    display["relative_exposure_x"] = display["relative_exposure"].round(2)
    display["queue"] = display["priority_tier"].map(lambda value: _localized_queue(str(value), lang))
    display_columns = [
        "rank",
        "flight",
        "route",
        "scheduled_departure",
        "delay_probability_pct",
        "route_rate_pct",
        "relative_exposure_x",
        "route_support",
        "schedule_percentile",
        "queue",
    ]
    labels = {
        "en": ["Rank", "Flight", "Route", "Departure", "Probability", "Route rate", "Exposure", "Support", "Schedule pct.", "Queue"],
        "es": ["Pos.", "Vuelo", "Ruta", "Salida", "Probabilidad", "Tasa ruta", "Exposición", "Soporte", "Percentil", "Cola"],
    }[lang]
    display_table = display[display_columns].copy()
    display_table.columns = labels
    st.dataframe(display_table, width="stretch", hide_index=True, height=min(620, 84 + total * 35))

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        st.markdown(f"**{t['batch']['distribution']}**")
        distribution = ranked["priority_tier"].value_counts().reindex(["Priority", "Watch", "Routine"]).fillna(0)
        distribution.index = [_localized_queue(str(value), lang) for value in distribution.index]
        st.bar_chart(distribution, height=250)
    with right:
        st.markdown('<div class="fr-panel">', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        s1.metric(t["batch"]["average"], _fmt_pct(avg))
        s2.metric(t["batch"]["unseen"], int((~ranked["route_seen"].astype(bool)).sum()))
        s3, s4 = st.columns(2)
        s3.metric(t["batch"]["low_support"], int(ranked["low_support"].sum()))
        s4.metric(t["metrics"]["calibration"], str(ranked["calibration_method"].iloc[0]))
        st.markdown("</div>", unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    ranked.to_csv(csv_buffer, index=False)
    pdf = report_service.build_schedule_brief_pdf(ranked, lang=lang)
    d1, d2 = st.columns(2)
    d1.download_button(
        t["batch"]["download_csv"],
        data=csv_buffer.getvalue(),
        file_name="flightrisk_ranked_schedule.csv",
        mime="text/csv",
        width="stretch",
    )
    d2.download_button(
        t["batch"]["download_pdf"],
        data=pdf,
        file_name=f"flightrisk_ranked_schedule_{lang}.pdf",
        mime="application/pdf",
        width="stretch",
    )
    st.caption(t["batch"]["caption"])


# ---------------------------------------------------------------------------
# Validation evidence
# ---------------------------------------------------------------------------


def _model_metrics(report: dict[str, Any], key: str) -> dict[str, Any]:
    block = report.get(key, {}) if isinstance(report, dict) else {}
    return block.get("metrics", block) if isinstance(block, dict) else {}


def _render_validation(lang: str, t: dict[str, Any]) -> None:
    report = _load_json(REPORTS_DIR / "metrics.json")
    backtest = _load_json(REPORTS_DIR / "temporal_backtest.json")
    candidate_benchmark = _load_json(REPORTS_DIR / "candidate_benchmark.json")
    main_metrics = _model_metrics(report, "main_model")
    baseline_metrics = _model_metrics(report, "baseline_model")
    selected_name = report.get("main_model", {}).get("model_name", "unknown")
    baseline_name = report.get("baseline_model", {}).get("model_name", "logistic_regression")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["validation"]["heldout_pr"], f"{main_metrics.get('pr_auc', 0):.3f}")
    m2.metric(t["validation"]["lift"], f"{main_metrics.get('lift_at_top_10pct', 0):.2f}×")
    m3.metric(t["validation"]["brier"], f"{main_metrics.get('brier_score', 0):.3f}")
    m4.metric(t["validation"]["ece"], f"{main_metrics.get('expected_calibration_error', 0):.3f}")

    main_pr = float(main_metrics.get("pr_auc", 0) or 0)
    baseline_pr = float(baseline_metrics.get("pr_auc", 0) or 0)
    if lang == "es":
        note = (
            f"{t['validation']['honest']}: {selected_name} se mantiene como artefacto final por su estabilidad en 4/4 folds. "
            f"En test obtiene PR-AUC {main_pr:.3f}, frente a {baseline_pr:.3f} de {baseline_name}. La diferencia es pequeña y se conserva explícitamente."
        )
    else:
        note = (
            f"{t['validation']['honest']}: {selected_name} remains the final artifact because it was stable in 4/4 temporal folds. "
            f"On test it reaches PR-AUC {main_pr:.3f}, versus {baseline_pr:.3f} for {baseline_name}. The small difference is retained explicitly."
        )
    st.markdown(f'<div class="fr-note emphasis">{html.escape(note)}</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame(
        [
            {
                "Model": selected_name,
                "ROC-AUC": main_metrics.get("roc_auc"),
                "PR-AUC": main_metrics.get("pr_auc"),
                "Precision@10%": main_metrics.get("precision_at_top_10pct"),
                "Lift@10%": main_metrics.get("lift_at_top_10pct"),
                "Brier": main_metrics.get("brier_score"),
                "ECE": main_metrics.get("expected_calibration_error"),
            },
            {
                "Model": baseline_name,
                "ROC-AUC": baseline_metrics.get("roc_auc"),
                "PR-AUC": baseline_metrics.get("pr_auc"),
                "Precision@10%": baseline_metrics.get("precision_at_top_10pct"),
                "Lift@10%": baseline_metrics.get("lift_at_top_10pct"),
                "Brier": baseline_metrics.get("brier_score"),
                "ECE": baseline_metrics.get("expected_calibration_error"),
            },
        ]
    )
    st.markdown(f"### {t['validation']['comparison']}")
    st.dataframe(comparison, width="stretch", hide_index=True)

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        calibration = report.get("main_model", {}).get("calibration", {})
        predicted = calibration.get("mean_predicted_probability") or []
        observed = calibration.get("fraction_of_positives") or []
        if predicted and observed and len(predicted) == len(observed):
            calibration_df = pd.DataFrame(
                {"Observed": observed, "Perfect calibration": predicted},
                index=pd.Index(predicted, name="Mean predicted probability"),
            )
            st.markdown(f"**{t['validation']['reliability']}**")
            st.line_chart(calibration_df, height=300)
            st.caption(t["validation"]["reliability_caption"])
    with right:
        cards = [
            (t["validation"]["encoding_title"], t["validation"]["encoding_body"]),
            (t["validation"]["calibration_title"], t["validation"]["calibration_body"]),
            (t["validation"]["backtest_title"], t["validation"]["backtest_body"]),
        ]
        for title, body in cards:
            st.markdown(
                f'<div class="fr-validation-card"><b>{html.escape(str(title))}</b><span>{html.escape(str(body))}</span></div>',
                unsafe_allow_html=True,
            )
            st.write("")

    if backtest.get("folds"):
        st.markdown(f"### {t['validation']['stability']}")
        summary_metrics = backtest.get("summary", {}).get("metrics", {})
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(t["validation"]["mean_pr"], f"{summary_metrics.get('pr_auc', {}).get('mean', 0):.3f}")
        b2.metric(t["validation"]["mean_lift"], f"{summary_metrics.get('lift_at_top_10pct', {}).get('mean', 0):.2f}×")
        b3.metric(t["validation"]["mean_brier"], f"{summary_metrics.get('brier_score', {}).get('mean', 0):.3f}")
        b4.metric(t["validation"]["mean_ece"], f"{summary_metrics.get('expected_calibration_error', {}).get('mean', 0):.3f}")

        fold_rows = []
        for fold in backtest["folds"]:
            metrics = fold.get("metrics", {})
            fold_rows.append(
                {
                    "Fold": fold.get("fold"),
                    "Test period": f"{fold.get('test_start')} → {fold.get('test_end')}",
                    "Selected model": fold.get("selected_model"),
                    "Calibration": fold.get("calibration_method"),
                    "PR-AUC": metrics.get("pr_auc"),
                    "Lift@10%": metrics.get("lift_at_top_10pct"),
                    "Brier": metrics.get("brier_score"),
                    "ECE": metrics.get("expected_calibration_error"),
                }
            )
        fold_df = pd.DataFrame(fold_rows)
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown(f"**{t['validation']['ranking_chart']}**")
            st.line_chart(fold_df.set_index("Fold")[["PR-AUC"]], height=240)
        with chart_right:
            st.markdown(f"**{t['validation']['calibration_chart']}**")
            st.line_chart(fold_df.set_index("Fold")[["Brier", "ECE"]], height=240)
        st.dataframe(fold_df, width="stretch", hide_index=True)

    benchmark_metrics = candidate_benchmark.get("validation_metrics", {})
    if benchmark_metrics:
        rows = []
        for model_name, metrics in benchmark_metrics.items():
            rows.append(
                {
                    "Candidate": model_name,
                    "Validation PR-AUC": metrics.get("pr_auc"),
                    "Lift@10%": metrics.get("lift_at_top_10pct"),
                    "Raw Brier": metrics.get("brier_score"),
                    "Raw ECE": metrics.get("expected_calibration_error"),
                }
            )
        st.markdown(f"### {t['validation']['benchmark']}")
        st.caption(t["validation"]["benchmark_caption"])
        st.dataframe(pd.DataFrame(rows).sort_values("Validation PR-AUC", ascending=False), width="stretch", hide_index=True)

    calibration_candidates = report.get("calibration_selection", {}).get("selected_model_candidates", {})
    if calibration_candidates:
        calibration_rows = [
            {
                "Method": method,
                "Validation Brier": metrics.get("brier_score"),
                "Validation ECE": metrics.get("expected_calibration_error"),
                "Validation log loss": metrics.get("log_loss"),
            }
            for method, metrics in calibration_candidates.items()
        ]
        st.markdown(f"### {t['validation']['calibration_candidates']}")
        st.dataframe(pd.DataFrame(calibration_rows).sort_values("Validation Brier"), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Model, monitoring and operations
# ---------------------------------------------------------------------------


def _render_operations(info: dict[str, Any], card: dict[str, Any], lang: str, t: dict[str, Any]) -> None:
    metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
    main = metrics.get("main_model", {}) if isinstance(metrics, dict) else {}
    main_metrics = main.get("metrics", main) if isinstance(main, dict) else {}
    performance = _load_json(REPORTS_DIR / "performance_benchmark.json")
    monitoring = prediction_service.prediction_summary()
    drift = prediction_service.drift_summary()

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="fr-validation-card"><span>{t["operations"]["model"]}</span><b>{html.escape(str(card.get("selected_model", info.get("model_name", "unknown"))))}</b></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="fr-validation-card"><span>{t["operations"]["rows"]}</span><b>{_fmt_int(info.get("n_train_rows"))}</b></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="fr-validation-card"><span>{t["operations"]["features"]}</span><b>{len(info.get("feature_columns") or [])}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t['operations']['monitoring']}")
    if int(monitoring.get("total_predictions", 0) or 0) == 0:
        st.markdown(f'<div class="fr-note">{t["operations"]["no_traffic"]}</div>', unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns(4)
    o1.metric(t["operations"]["predictions"], int(monitoring.get("total_predictions", 0) or 0))
    o2.metric(t["operations"]["average"], _fmt_pct(monitoring.get("average_probability")))
    o3.metric(t["operations"]["drift"], str(drift.get("status", "n/a")).upper())
    o4.metric(t["operations"]["latest"], str(monitoring.get("latest_prediction_utc") or "n/a")[:19])
    if drift.get("features"):
        drift_df = pd.DataFrame(
            [{"Feature": key, "PSI": value} for key, value in drift.get("features", {}).items()]
        ).sort_values("PSI", ascending=False)
        st.dataframe(drift_df, width="stretch", hide_index=True)

    st.markdown(f"### {t['operations']['performance']}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(t["operations"]["artifact_load"], _fmt_ms(performance.get("artifact_load_ms")))
    p2.metric(t["operations"]["single_latency"], _fmt_ms(performance.get("single_prediction_ms", {}).get("median")))
    p3.metric(t["operations"]["batch_100"], _fmt_ms(performance.get("batch_100_ms", {}).get("median")))
    p4.metric(t["operations"]["batch_1000"], _fmt_ms(performance.get("batch_1000_ms", {}).get("median")))
    st.caption(t["operations"]["environment"])

    with st.expander(t["operations"]["model_card"], expanded=True):
        intended = card.get("intended_use", "Portfolio ML evaluation")
        not_intended = card.get("not_intended_use", "Operational aviation decisions")
        if lang == "es":
            intended = "Sistema educativo de portfolio para estimar riesgo de retraso con información del horario."
            not_intended = "Aviación operacional, seguridad, despacho o decisiones de viaje de alto impacto."
        st.markdown(
            f"""
- **Task / Tarea:** {card.get('task', 'Binary arrival-delay classification')}
- **Target:** `{card.get('target', 'ArrDel15')}`
- **Release:** `v{APP_VERSION} · {RELEASE_NAME}`
- **Artifact:** `{info.get('version', 'unknown')}`
- **Calibration:** `{info.get('calibration_method', 'identity')}`
- **Historical encoding:** `{info.get('historical_encoding', 'unknown')}`
- **Held-out PR-AUC:** `{main_metrics.get('pr_auc', 'n/a')}`
- **Held-out Brier:** `{main_metrics.get('brier_score', 'n/a')}`
- **Intended use:** {intended}
- **Not intended for:** {not_intended}
"""
        )

    with st.expander(t["operations"]["leakage"]):
        if lang == "es":
            st.markdown(
                """
**Permitido antes de la salida**

- aerolínea, origen y destino
- calendario y horas programadas
- duración programada y distancia
- agregados históricos ordenados construidos con fechas estrictamente anteriores

**Bloqueado explícitamente**

- `ArrDelay`, `DepDelay`, `ArrDelayMinutes`
- salida/llegada real, taxi y wheels times
- duración real y tiempo en el aire
- causas de retraso de aerolínea, meteorología, NAS o aeronave anterior
- cancelación y desvío como variables de inferencia
"""
            )
        else:
            st.markdown(
                """
**Allowed before departure**

- carrier, origin and destination
- calendar and scheduled times
- scheduled duration and distance
- ordered historical aggregates built from strictly earlier dates

**Explicitly blocked**

- `ArrDelay`, `DepDelay`, `ArrDelayMinutes`
- actual departure/arrival, taxi and wheels times
- actual elapsed time and airborne time
- carrier, weather, NAS and late-aircraft delay causes
- cancellation and diversion status as inference features
"""
            )

    with st.expander(t["operations"]["api"]):
        st.code(
            """GET  /health
GET  /model/info
GET  /model/card
POST /predict
POST /predict/batch
POST /rank
POST /reports/flight
POST /reports/schedule
GET  /monitoring/summary
GET  /monitoring/drift""",
            language="text",
        )
        st.markdown("FastAPI OpenAPI: `/docs`")

    with st.expander(t["operations"]["architecture"]):
        st.code(
            """app/api/           FastAPI transport layer
app/dashboard/     bilingual Streamlit product surface
app/services/      inference and PDF report services
src/data/          loading, cleaning, temporal splitting
src/features/      schedule features and historical aggregates
src/models/        training, calibration, explanation, inference
src/monitoring/    prediction logging and PSI drift checks
scripts/           reproducible CLI and benchmark workflows
reports/           committed evaluation and performance evidence""",
            language="text",
        )

    with st.expander(t["operations"]["deployment"]):
        st.markdown(t["operations"]["deployment_body"])
        st.code(
            """docker compose up --build
streamlit run app/dashboard/streamlit_app.py
uvicorn app.api.main:app --host 0.0.0.0 --port 8000""",
            language="bash",
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    _inject_theme()
    info = _safe_model_info()
    card = _safe_model_card()
    model_available = prediction_service.is_model_available()

    top_left, top_right = st.columns([0.84, 0.16], gap="small")
    with top_right:
        lang, t = _language_selector()
    with top_left:
        _topbar(model_available, info.get("version"), t)

    _hero(t, model_available)

    tabs = st.tabs(t["tabs"])
    with tabs[0]:
        _section_header(t["analyze_title"], t["analyze_sub"])
        form_result = _flight_form(lang, t, _safe_catalog(), disabled=not model_available)
        if form_result is not None:
            payload, metadata = form_result
            _render_prediction(payload, metadata, lang, t)
        else:
            st.markdown(f'<div class="fr-note">{t["single"]["idle"]}</div>', unsafe_allow_html=True)

    with tabs[1]:
        _section_header(t["rank_title"], t["rank_sub"])
        _render_batch(model_available, lang, t)

    with tabs[2]:
        _section_header(t["validation_title"], t["validation_sub"])
        _render_validation(lang, t)

    with tabs[3]:
        _section_header(t["operations_title"], t["operations_sub"])
        _render_operations(info, card, lang, t)

    st.markdown(f'<div class="fr-footer">{t["footer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
