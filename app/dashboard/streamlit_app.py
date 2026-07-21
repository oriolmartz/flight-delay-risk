"""Flight Delay Risk public product dashboard.

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
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.dashboard.i18n import TEXT
from app.dashboard.theme import CSS
from app.services import prediction_service, report_service
from src.models.predict import PredictionInput
from src.version import APP_VERSION, RELEASE_NAME

st.set_page_config(
    page_title="Flight Delay Risk - Pre-departure Risk Workbench",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports"
AIRPORT_MAP_PATH = ROOT / "docs" / "assets" / "us_airport_coverage.svg"
TRAINING_SOURCE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_schedule.csv"
TEMPLATE_PATH = ROOT / "data" / "sample" / "schedule_template.csv"
MAX_UPLOAD_ROWS = 500
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG_NAMESPACES = {"svg": SVG_NAMESPACE}
ET.register_namespace("", SVG_NAMESPACE)


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


def _safe_airport_history() -> list[dict[str, Any]]:
    try:
        return prediction_service.airport_historical_summary()
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _airport_map_source() -> str:
    return AIRPORT_MAP_PATH.read_text(encoding="utf-8")


def _airport_code(circle: ET.Element) -> str | None:
    title = circle.find(f"{{{SVG_NAMESPACE}}}title")
    if title is None or not title.text:
        return None
    return title.text.split(" — ", 1)[0].strip()


def _coverage_map_svg(airports: list[str], aria_label: str) -> str:
    """Render only airport points exposed by the loaded artifact."""
    try:
        root = ET.fromstring(_airport_map_source())
    except (OSError, ET.ParseError):
        return ""
    supported = set(airports)
    airport_group = root.find(".//svg:g[@class='fr-airports']", SVG_NAMESPACES)
    if airport_group is not None and supported:
        for circle in list(airport_group):
            code = _airport_code(circle)
            if code and code not in supported:
                airport_group.remove(circle)
    label_group = root.find(".//svg:g[@class='fr-airport-labels']", SVG_NAMESPACES)
    if label_group is not None and supported:
        for label in list(label_group):
            if (label.text or "").strip() not in supported:
                label_group.remove(label)
    root.set("aria-label", aria_label)
    return ET.tostring(root, encoding="unicode")


def _heatmap_color(rate: float) -> str:
    if rate < 0.18:
        return "#53c7f0"
    if rate < 0.21:
        return "#a1dcf2"
    if rate < 0.24:
        return "#f2b84b"
    return "#ff6b5e"


def _airport_heatmap_svg(
    history: list[dict[str, Any]],
    mode: str,
    aria_label: str,
    tooltip_rate_template: str,
    tooltip_support_template: str,
) -> str:
    """Build an artifact-backed proportional-symbol heatmap on the U.S. SVG."""
    try:
        root = ET.fromstring(_airport_map_source())
    except (OSError, ET.ParseError):
        return ""
    root.set("class", "fr-heatmap-svg")
    root.set("aria-label", aria_label)
    rows = {str(row.get("airport")): row for row in history if row.get("airport")}
    rate_key = "origin_rate" if mode == "origin" else "destination_rate"
    support_key = "origin_support" if mode == "origin" else "destination_support"
    support_logs = [math.log1p(max(int(row.get(support_key, 0) or 0), 0)) for row in rows.values()]
    low_log = min(support_logs, default=0.0)
    high_log = max(support_logs, default=1.0)
    log_span = max(high_log - low_log, 1e-9)
    airport_group = root.find(".//svg:g[@class='fr-airports']", SVG_NAMESPACES)
    tooltip_layer = ET.Element(
        f"{{{SVG_NAMESPACE}}}g",
        {"class": "fr-heat-tooltip-layer", "aria-hidden": "true"},
    )
    root.append(tooltip_layer)
    if airport_group is not None:
        for circle in list(airport_group):
            code = _airport_code(circle)
            row = rows.get(code or "")
            if row is None:
                airport_group.remove(circle)
                continue
            rate = float(row.get(rate_key, 0.0) or 0.0)
            support = max(int(row.get(support_key, 0) or 0), 0)
            normalized_support = (math.log1p(support) - low_log) / log_span
            radius = 1.8 + 4.3 * math.sqrt(max(0.0, min(normalized_support, 1.0)))
            circle.set("class", "fr-heat-dot")
            circle.set("r", f"{radius:.2f}")
            circle.set("fill", _heatmap_color(rate))
            circle.set("fill-opacity", ".9")
            circle.set("stroke", "#ffffff")
            circle.set("stroke-opacity", ".96")
            circle.set("stroke-width", ".82")
            rate_line = tooltip_rate_template.format(rate=f"{rate:.1%}")
            support_line = tooltip_support_template.format(support=f"{support:,}")
            accessible_label = f"{code} · {rate_line} · {support_line}"
            circle.set("aria-label", accessible_label)
            title = circle.find(f"{{{SVG_NAMESPACE}}}title")
            if title is not None:
                circle.remove(title)

            cx = float(circle.get("cx", "0"))
            cy = float(circle.get("cy", "0"))
            tooltip_width = 220.0
            tooltip_height = 59.0
            tooltip_x = cx + 12.0
            if tooltip_x + tooltip_width > 967.0:
                tooltip_x = cx - tooltip_width - 12.0
            tooltip_x = max(8.0, min(tooltip_x, 967.0 - tooltip_width))
            tooltip_y = cy - tooltip_height - 12.0
            if tooltip_y < 8.0:
                tooltip_y = cy + 12.0
            tooltip_y = max(8.0, min(tooltip_y, 602.0 - tooltip_height))

            hover_group = ET.SubElement(
                tooltip_layer,
                f"{{{SVG_NAMESPACE}}}g",
                {"class": "fr-heat-hover", "aria-label": accessible_label},
            )
            ET.SubElement(
                hover_group,
                f"{{{SVG_NAMESPACE}}}circle",
                {
                    "class": "fr-heat-hit",
                    "cx": f"{cx:.1f}",
                    "cy": f"{cy:.1f}",
                    "r": f"{radius + 3.2:.2f}",
                    "fill": "none",
                    "stroke": "none",
                    "pointer-events": "all",
                },
            )
            tooltip = ET.SubElement(
                hover_group,
                f"{{{SVG_NAMESPACE}}}g",
                {
                    "class": "fr-heat-tooltip",
                    "aria-hidden": "true",
                    "transform": f"translate({tooltip_x:.1f} {tooltip_y:.1f})",
                },
            )
            ET.SubElement(
                tooltip,
                f"{{{SVG_NAMESPACE}}}rect",
                {"width": str(tooltip_width), "height": str(tooltip_height), "rx": "7"},
            )
            code_text = ET.SubElement(
                tooltip,
                f"{{{SVG_NAMESPACE}}}text",
                {"class": "fr-heat-tooltip-code", "x": "12", "y": "18"},
            )
            code_text.text = str(code)
            rate_text = ET.SubElement(
                tooltip,
                f"{{{SVG_NAMESPACE}}}text",
                {"class": "fr-heat-tooltip-rate", "x": "12", "y": "36"},
            )
            rate_text.text = rate_line
            support_text = ET.SubElement(
                tooltip,
                f"{{{SVG_NAMESPACE}}}text",
                {"class": "fr-heat-tooltip-support", "x": "12", "y": "51"},
            )
            support_text.text = support_line
    root.set("data-mode", mode)
    return ET.tostring(root, encoding="unicode")


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
        "calibration_method": "sigmoid",
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


def _support_quality(value: Any, t: dict[str, Any]) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if value < 100:
        return str(t["single"]["support_low"])
    if value < 500:
        return str(t["single"]["support_moderate"])
    return str(t["single"]["support_high"])


def _model_display_name(value: Any) -> str:
    labels = {
        "baseline": "Logistic Regression",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "mlp_embeddings": "MLP embeddings",
        "ft_transformer": "FT-Transformer",
    }
    key = str(value or "unknown")
    return labels.get(key, key.replace("_", " ").title())


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


def _metric_cards(cards: list[dict[str, str]], columns: int = 3) -> None:
    """Render compact, self-explanatory metric cards without hiding meaning in tooltips."""
    blocks = []
    for card in cards:
        direction = card.get("direction", "")
        direction_html = (
            f'<div class="fr-metric-direction">{html.escape(direction)}</div>' if direction else ""
        )
        blocks.append(
            f"""<div class="fr-metric-card">
  <div class="fr-metric-label">{html.escape(card["label"])}</div>
  <div class="fr-metric-value">{html.escape(card["value"])}</div>
  <div class="fr-metric-help">{html.escape(card["help"])}</div>
  {direction_html}
</div>"""
        )
    st.markdown(
        f'<div class="fr-metric-grid cols-{max(1, min(columns, 4))}">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def _relative_help(relative: float, t: dict[str, Any]) -> str:
    if relative > 1.02:
        return str(t["metric_help"]["relative_above"]).format(delta=round((relative - 1) * 100))
    if relative < 0.98:
        return str(t["metric_help"]["relative_below"]).format(delta=round((1 - relative) * 100))
    return str(t["metric_help"]["relative_equal"])


def _drift_explanation(status: Any, t: dict[str, Any]) -> str:
    normalized = str(status or "low").lower()
    if normalized in {"high", "critical"}:
        return str(t["operations"]["drift_plain_high"])
    if normalized in {"medium", "moderate", "warning"}:
        return str(t["operations"]["drift_plain_medium"])
    return str(t["operations"]["drift_plain_low"])


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
      <div class="fr-brand-name">FLIGHT DELAY RISK</div>
      <div class="fr-byline">{html.escape(str(t['topbar']['byline']))}</div>
      <a class="fr-source-link" href="{TRAINING_SOURCE_URL}" target="_blank" rel="noopener noreferrer">
        {html.escape(str(t['topbar']['source']))}<span aria-hidden="true">↗</span>
      </a>
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


def _hero(
    t: dict[str, Any],
    model_available: bool,
    catalog: dict[str, list[str]],
) -> None:
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
    airport_count = len(catalog.get("airports", []))
    map_count = str(t["hero_map"]["count"]).format(count=airport_count)
    map_aria = str(t["hero_map"]["aria"]).format(count=airport_count)
    coverage_svg = _coverage_map_svg(catalog.get("airports", []), map_aria)
    st.markdown(
        f"""
<section class="fr-hero">
  <div class="fr-hero-copy">
    <div class="fr-kicker">{t['hero_kicker']}</div>
    <div class="fr-title">{t['hero_title']}</div>
    <div class="fr-subtitle">{t['hero_sub']}</div>
    <div class="fr-constraint">{t['constraint']}</div>
  </div>
  <div class="fr-coverage-visual">
    <div class="fr-coverage-head">
      <div>
        <span>{html.escape(str(t['hero_map']['eyebrow']))}</span>
        <strong>{html.escape(map_count)}</strong>
      </div>
      <p>{html.escape(str(t['hero_map']['caption']))}</p>
    </div>
    <div class="fr-map-frame">{coverage_svg}</div>
    <div class="fr-flight-card">
      <div class="fr-flight-top">
        <div>
          <div class="fr-flight-id">{t['hero_card']['example']}</div>
          <div class="fr-route">{html.escape(str(route))}</div>
        </div>
        <div class="fr-priority">{html.escape(str(priority))}</div>
      </div>
      <div class="fr-risk-row">
        <div>
          <div class="fr-risk-number">{_fmt_pct(probability)}</div>
          <div class="fr-risk-label">{t['hero_card']['probability']}</div>
        </div>
        <div class="fr-flight-grid">
          <div class="fr-flight-stat"><b>{_fmt_pct(route_rate)}</b><span>{t['hero_card']['route_rate']}</span></div>
          <div class="fr-flight-stat"><b>{relative:.2f}×</b><span>{t['hero_card']['relative']}</span></div>
          <div class="fr-flight-stat"><b>{_fmt_int(support)} · {_support_quality(support, t)}</b><span>{t['hero_card']['support']}</span></div>
        </div>
      </div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )
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
        flight_date=flight_date.isoformat(),
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

    for item in contributions[:5]:
        contribution = float(item.get("contribution", 0.0))
        direction_class = "up" if contribution >= 0 else "down"
        direction_label = t["single"]["increase"] if contribution >= 0 else t["single"]["decrease"]
        label, raw_label = _contribution_label(item, t)
        st.markdown(
            f"""
<div class="fr-contribution">
  <div><b>{html.escape(label)}</b><span>{html.escape(raw_label)}</span></div>
  <div class="fr-contribution-value {direction_class}">{html.escape(str(direction_label))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander(t["single"]["technical_explanation"]):
        technical_rows = []
        for item in contributions:
            label, raw_label = _contribution_label(item, t)
            technical_rows.append(
                {
                    "Feature": label,
                    "Observed value": raw_label,
                    "Log-odds contribution": float(item.get("contribution", 0.0)),
                    "Direction": item.get("direction", ""),
                }
            )
        st.dataframe(pd.DataFrame(technical_rows), width="stretch", hide_index=True)


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

    route_support = context.get("route_support")
    support_count = _fmt_int(route_support)
    support_quality = _support_quality(route_support, t)
    _metric_cards(
        [
            {
                "label": str(t["metrics"]["probability"]),
                "value": _fmt_pct(probability),
                "help": str(t["metric_help"]["probability"]),
            },
            {
                "label": str(t["metrics"]["relative"]),
                "value": f"{relative:.2f}×",
                "help": _relative_help(relative, t),
            },
            {
                "label": str(t["metrics"]["support"]),
                "value": f"{support_count} · {support_quality}",
                "help": str(t["metric_help"]["support"]).format(count=support_count),
            },
        ],
        columns=3,
    )
    st.markdown(
        f'<div class="fr-context-summary">{html.escape(str(t["single"]["route_summary"]).format(route_rate=_fmt_pct(route_rate), global_rate=_fmt_pct(global_rate)))}</div>',
        unsafe_allow_html=True,
    )

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
  <div class="fr-reliability-callout">{html.escape(str(t['single']['support_summary']).format(count=support_count, quality=support_quality.lower()))}</div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['coverage']}</div><div class="fr-context-value">{route_seen}</div></div>
  <div class="fr-context-row"><div class="fr-context-label">{t['metrics']['carrier_route_coverage']}</div><div class="fr-context-value">{carrier_route_seen}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    _render_contributions(result, t)

    with st.expander(t["single"]["advanced_details"]):
        st.caption(t["single"]["advanced_details_note"])
        advanced = pd.DataFrame(
            [
                {"Field": t["metrics"]["probability"], "Value": _fmt_pct(probability), "Meaning": t["metric_help"]["probability"]},
                {"Field": t["metrics"]["raw_score"], "Value": _fmt_pct(raw_score), "Meaning": "Model output before probability calibration." if lang == "en" else "Salida del modelo antes de calibrar la probabilidad."},
                {"Field": t["metrics"]["calibration"], "Value": calibration_method, "Meaning": "Method used to turn the raw score into a more realistic probability." if lang == "en" else "Método que convierte el score bruto en una probabilidad más realista."},
                {"Field": t["metrics"]["fallback"], "Value": _fmt_pct(global_rate), "Meaning": t["metric_help"]["prevalence"]},
            ]
        )
        st.dataframe(advanced, width="stretch", hide_index=True)

    pdf = report_service.build_flight_brief_pdf(metadata, result, context, lang=lang)
    filename = f"flight_delay_risk_{metadata['airline']}_{metadata.get('flight_number') or 'flight'}_{lang}.pdf"
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
                    flight_date=date_label if date_label else None,
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

    unseen_pre = 0
    low_support_pre = 0
    if validated.payloads:
        contexts = prediction_service.prediction_contexts(validated.payloads)
        unseen_pre = sum(not bool(item.get("route_seen")) for item in contexts)
        low_support_pre = sum(int(item.get("route_support", 0)) < 100 for item in contexts)
    evidence_gaps = unseen_pre + low_support_pre
    _metric_cards(
        [
            {"label": str(t["batch"]["valid"]), "value": str(len(validated.prepared)), "help": str(t["metric_help"]["valid_rows"])},
            {"label": str(t["batch"]["invalid"]), "value": str(len(validated.errors)), "help": str(t["metric_help"]["invalid_rows"])},
            {"label": str(t["batch"]["evidence_gaps"]), "value": str(evidence_gaps), "help": str(t["metric_help"]["evidence_gaps"])},
        ],
        columns=3,
    )

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

    st.markdown(f'<div class="fr-note emphasis">{html.escape(str(t["batch"]["summary_intro"]))}</div>', unsafe_allow_html=True)
    _metric_cards(
        [
            {"label": str(t["batch"]["priority"]), "value": str(priority), "help": str(t["metric_help"]["priority_queue"])},
            {"label": str(t["batch"]["watch"]), "value": str(watch), "help": str(t["metric_help"]["watch_queue"])},
            {"label": str(t["batch"]["highest"]), "value": _fmt_pct(max_probability), "help": str(t["metric_help"]["highest"])},
        ],
        columns=3,
    )

    st.markdown(f"### {t['batch']['ranked']}")
    st.caption(t["batch"]["table_help"])
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
        "en": ["Rank", "Flight", "Route", "Departure", "Probability", "Route rate", "Risk vs route", "Support", "Schedule pct.", "Queue"],
        "es": ["Pos.", "Vuelo", "Ruta", "Salida", "Probabilidad", "Tasa ruta", "Riesgo vs ruta", "Soporte", "Percentil", "Cola"],
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
        _metric_cards(
            [
                {"label": str(t["batch"]["average"]), "value": _fmt_pct(avg), "help": str(t["metric_help"]["average"])},
                {"label": str(t["batch"]["unseen"]), "value": str(int((~ranked["route_seen"].astype(bool)).sum())), "help": str(t["metric_help"]["evidence_gaps"])},
            ],
            columns=2,
        )
        with st.expander(t["batch"]["advanced_summary"]):
            st.write({
                t["batch"]["low_support"]: int(ranked["low_support"].sum()),
                t["metrics"]["calibration"]: str(ranked["calibration_method"].iloc[0]),
                t["batch"]["valid"]: total,
            })

    csv_buffer = io.StringIO()
    ranked.to_csv(csv_buffer, index=False)
    pdf = report_service.build_schedule_brief_pdf(ranked, lang=lang)
    d1, d2 = st.columns(2)
    d1.download_button(
        t["batch"]["download_csv"],
        data=csv_buffer.getvalue(),
        file_name="flight_delay_risk_ranked_schedule.csv",
        mime="text/csv",
        width="stretch",
    )
    d2.download_button(
        t["batch"]["download_pdf"],
        data=pdf,
        file_name=f"flight_delay_risk_ranked_schedule_{lang}.pdf",
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


def _render_airport_heatmap(lang: str, t: dict[str, Any]) -> None:
    history = _safe_airport_history()
    if not history:
        return
    copy = t["heatmap"]
    mode = st.segmented_control(
        str(copy["mode"]),
        options=("origin", "destination"),
        format_func=lambda value: str(
            copy["origin" if value == "origin" else "destination"]
        ),
        default="origin",
        selection_mode="single",
        label_visibility="collapsed",
        key=f"airport_heatmap_mode_{lang}",
    )
    mode = str(mode or "origin")
    aria_key = "aria_origin" if mode == "origin" else "aria_destination"
    heatmap_svg = _airport_heatmap_svg(
        history,
        mode,
        str(copy[aria_key]),
        str(copy["tooltip_rate"]),
        str(copy["tooltip_support"]),
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        for color, label in (
            ("#53c7f0", "<18%"),
            ("#a1dcf2", "18–21%"),
            ("#f2b84b", "21–24%"),
            ("#ff6b5e", "≥24%"),
        )
    )
    airport_count = str(copy["count"]).format(count=len(history))
    st.markdown(
        f"""
<div class="fr-heatmap-shell">
  <div class="fr-heatmap-intro">
    <div>
      <span class="fr-heatmap-eyebrow">{html.escape(str(copy['eyebrow']))}</span>
      <strong>{html.escape(str(copy['title']))}</strong>
      <p>{html.escape(str(copy['caption']))}</p>
    </div>
    <div class="fr-heatmap-meta">
      <span>{html.escape(airport_count)}</span>
      <span>{html.escape(str(copy['target']))}</span>
      <em>{html.escape(str(copy['disclaimer']))}</em>
    </div>
  </div>
  <div class="fr-heatmap-map">{heatmap_svg}</div>
  <div class="fr-heatmap-legend">
    <b>{html.escape(str(copy['legend_rate']))}</b>
    {legend}
    <em>{html.escape(str(copy['legend_support']))}</em>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_validation(lang: str, t: dict[str, Any]) -> None:
    report = _load_json(REPORTS_DIR / "metrics.json")
    backtest = _load_json(REPORTS_DIR / "temporal_backtest.json")
    candidate_benchmark = _load_json(REPORTS_DIR / "candidate_benchmark.json")
    main_metrics = _model_metrics(report, "main_model")
    baseline_metrics = _model_metrics(report, "baseline_model")
    selected_name = report.get("main_model", {}).get("model_name", "unknown")
    baseline_name = report.get("baseline_model", {}).get("model_name", "logistic_regression")

    main_pr = float(main_metrics.get("pr_auc", 0) or 0)
    prevalence = float(
        main_metrics.get("baseline_positive_rate")
        or main_metrics.get("positive_rate_actual")
        or main_metrics.get("observed_positive_rate")
        or 0
    )
    lift = float(main_metrics.get("lift_at_top_10pct", 0) or 0)
    brier = float(main_metrics.get("brier_score", 0) or 0)
    ece = float(main_metrics.get("expected_calibration_error", 0) or 0)
    ratio = main_pr / prevalence if prevalence else 0.0

    st.markdown(f"### {t['validation']['overview']}")
    st.markdown(f'<div class="fr-note">{html.escape(str(t["validation"]["overview_body"]))}</div>', unsafe_allow_html=True)
    _metric_cards(
        [
            {
                "label": str(t["validation"]["heldout_pr"]),
                "value": f"{main_pr:.3f}",
                "help": str(t["validation"]["ranking_plain"]).format(value=main_pr, prevalence=prevalence, ratio=ratio),
                "direction": str(t["validation"]["higher_better"]),
            },
            {
                "label": str(t["validation"]["lift"]),
                "value": f"{lift:.2f}×",
                "help": str(t["validation"]["lift_plain"]).format(extra=max(0.0, (lift - 1) * 100)),
                "direction": str(t["validation"]["higher_better"]),
            },
            {
                "label": str(t["validation"]["ece"]),
                "value": f"{ece * 100:.1f} pp",
                "help": str(t["validation"]["calibration_plain"]).format(points=ece * 100),
                "direction": str(t["validation"]["lower_better"]),
            },
        ],
        columns=3,
    )

    with st.expander(t["validation"]["metric_guide"]):
        st.markdown(t["validation"]["metric_guide_body"])

    main_baseline_pr = float(baseline_metrics.get("pr_auc", 0) or 0)
    backtest_summary = backtest.get("summary", {}) if isinstance(backtest, dict) else {}
    fold_count = int(backtest_summary.get("folds", 0) or 0)
    selected_counts = backtest_summary.get("selected_models", {}) or {}
    stability = ", ".join(
        f"{_model_display_name(name)}: {count}/{fold_count}"
        for name, count in sorted(selected_counts.items())
    ) or "no fold summary"
    if lang == "es":
        note = (
            f"{_model_display_name(selected_name)} se eligió antes del test final. En el test obtiene una calidad de ranking "
            f"de {main_pr:.3f}, frente a {main_baseline_pr:.3f} del baseline. Los periodos temporales no tienen un ganador único ({stability})."
        )
    else:
        note = (
            f"{_model_display_name(selected_name)} was selected before the final test. It reaches {main_pr:.3f} ranking quality "
            f"versus {main_baseline_pr:.3f} for the simple baseline. Temporal periods do not share one universal winner ({stability})."
        )
    st.markdown(f'<div class="fr-note emphasis">{html.escape(note)}</div>', unsafe_allow_html=True)

    comparison = pd.DataFrame(
        [
            {
                t["validation"]["model"]: _model_display_name(selected_name),
                t["validation"]["heldout_pr"]: main_metrics.get("pr_auc"),
                t["validation"]["priority_precision"]: main_metrics.get("precision_at_top_10pct"),
                t["validation"]["lift"]: main_metrics.get("lift_at_top_10pct"),
            },
            {
                t["validation"]["model"]: _model_display_name(baseline_name),
                t["validation"]["heldout_pr"]: baseline_metrics.get("pr_auc"),
                t["validation"]["priority_precision"]: baseline_metrics.get("precision_at_top_10pct"),
                t["validation"]["lift"]: baseline_metrics.get("lift_at_top_10pct"),
            },
        ]
    )
    st.markdown(f"### {t['validation']['comparison']}")
    st.caption(t["validation"]["comparison_caption"])
    st.dataframe(comparison, width="stretch", hide_index=True)

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        calibration = report.get("main_model", {}).get("calibration", {})
        predicted = calibration.get("mean_predicted_probability") or []
        observed = calibration.get("fraction_of_positives") or []
        if predicted and observed and len(predicted) == len(observed):
            calibration_df = pd.DataFrame(
                {"Observed outcome rate": observed, "Perfect probability match": predicted},
                index=pd.Index(predicted, name="Predicted probability"),
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
        st.markdown(f'<div class="fr-note">{html.escape(str(t["validation"]["stability_intro"]))}</div>', unsafe_allow_html=True)
        summary_metrics = backtest.get("summary", {}).get("metrics", {})
        mean_pr = float(summary_metrics.get("pr_auc", {}).get("mean", 0) or 0)
        mean_lift = float(summary_metrics.get("lift_at_top_10pct", {}).get("mean", 0) or 0)
        _metric_cards(
            [
                {"label": str(t["validation"]["mean_pr"]), "value": f"{mean_pr:.3f}", "help": str(t["metric_help"]["ranking"]), "direction": str(t["validation"]["higher_better"])},
                {"label": str(t["validation"]["mean_lift"]), "value": f"{mean_lift:.2f}×", "help": str(t["metric_help"]["lift"]), "direction": str(t["validation"]["higher_better"])},
            ],
            columns=2,
        )

        fold_rows = []
        badges = []
        for fold in backtest["folds"]:
            metrics = fold.get("metrics", {})
            fold_prevalence = float(
                metrics.get("baseline_positive_rate")
                or metrics.get("positive_rate_actual")
                or metrics.get("observed_positive_rate")
                or 0.0
            )
            pr_auc = float(metrics.get("pr_auc", 0.0) or 0.0)
            model_label = _model_display_name(fold.get("selected_model"))
            fold_rows.append(
                {
                    "Fold": int(fold.get("fold", 0)),
                    str(t["validation"]["period"]): f"{fold.get('test_start')} → {fold.get('test_end')}",
                    str(t["validation"]["selected_model"]): model_label,
                    str(t["validation"]["heldout_pr"]): pr_auc,
                    str(t["validation"]["delay_rate"]): fold_prevalence,
                    str(t["validation"]["lift"]): metrics.get("lift_at_top_10pct"),
                    "Calibration": str(fold.get("calibration_method", "unknown")).title(),
                    "Brier": metrics.get("brier_score"),
                    "ECE": metrics.get("expected_calibration_error"),
                    "PR-AUC / prevalence": pr_auc / fold_prevalence if fold_prevalence > 0 else None,
                }
            )
            badges.append(f'<span class="fr-model-badge"><b>Fold {fold.get("fold")}</b> · {html.escape(model_label)}</span>')
        st.markdown(
            f'<div class="fr-note">{html.escape(str(t["validation"]["no_dominant_model"]))}</div>'
            f'<div class="fr-model-badges">{"".join(badges)}</div>',
            unsafe_allow_html=True,
        )
        fold_df = pd.DataFrame(fold_rows)
        chart_df = fold_df[["Fold", str(t["validation"]["heldout_pr"]), str(t["validation"]["delay_rate"])]].melt(
            "Fold", var_name="Metric", value_name="Value"
        )
        ranking_chart = (
            alt.Chart(chart_df)
            .mark_line(point=alt.OverlayMarkDef(size=75))
            .encode(
                x=alt.X("Fold:O", title="Temporal fold"),
                y=alt.Y("Value:Q", title="Share / score", scale=alt.Scale(zero=False)),
                color=alt.Color("Metric:N", title=None),
                tooltip=["Fold:O", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
            )
            .properties(height=260)
        )
        labels = (
            alt.Chart(chart_df)
            .mark_text(dy=-12, fontSize=11)
            .encode(x="Fold:O", y="Value:Q", text=alt.Text("Value:Q", format=".3f"), color="Metric:N")
        )
        st.markdown(f"**{t['validation']['ranking_chart']}**")
        st.altair_chart(ranking_chart + labels, width="stretch")
        public_fold_columns = [
            "Fold", str(t["validation"]["period"]), str(t["validation"]["selected_model"]),
            str(t["validation"]["heldout_pr"]), str(t["validation"]["delay_rate"]), str(t["validation"]["lift"]),
        ]
        st.dataframe(fold_df[public_fold_columns], width="stretch", hide_index=True)

        with st.expander(t["validation"]["advanced_fold_table"]):
            calibration_fold_df = fold_df[["Fold", "Brier", "ECE"]].melt(
                "Fold", var_name="Metric", value_name="Value"
            )
            calibration_chart = (
                alt.Chart(calibration_fold_df)
                .mark_line(point=alt.OverlayMarkDef(size=75))
                .encode(
                    x=alt.X("Fold:O", title="Temporal fold"),
                    y=alt.Y("Value:Q", title="Error", scale=alt.Scale(zero=False)),
                    color=alt.Color("Metric:N", title=None),
                    tooltip=["Fold:O", "Metric:N", alt.Tooltip("Value:Q", format=".3f")],
                )
                .properties(height=235)
            )
            st.altair_chart(calibration_chart, width="stretch")
            st.dataframe(fold_df, width="stretch", hide_index=True)

    benchmark_metrics = candidate_benchmark.get("candidates", candidate_benchmark.get("validation_metrics", {}))
    allowed_candidates = [
        "baseline", "random_forest", "extra_trees", "xgboost", "lightgbm", "mlp_embeddings", "ft_transformer",
    ]
    if benchmark_metrics:
        rows = []
        for model_key in allowed_candidates:
            metrics = benchmark_metrics.get(model_key)
            if not metrics:
                continue
            rows.append(
                {
                    str(t["validation"]["model"]): _model_display_name(model_key),
                    str(t["validation"]["heldout_pr"]): metrics.get("pr_auc"),
                    str(t["validation"]["lift"]): metrics.get("lift_at_top_10pct"),
                    "Brier": metrics.get("brier_score"),
                    "ECE": metrics.get("expected_calibration_error"),
                }
            )
        benchmark_df = pd.DataFrame(rows).sort_values(str(t["validation"]["heldout_pr"]), ascending=False)
        st.markdown(f"### {t['validation']['benchmark']}")
        st.caption(t["validation"]["benchmark_caption"])
        st.dataframe(
            benchmark_df[[str(t["validation"]["model"]), str(t["validation"]["heldout_pr"]), str(t["validation"]["lift"])]],
            width="stretch", hide_index=True,
        )
        with st.expander(t["validation"]["advanced_model_table"]):
            st.dataframe(benchmark_df, width="stretch", hide_index=True)

    calibration_candidates = report.get("calibration_selection", {}).get("selected_model_candidates", {})
    with st.expander(t["validation"]["advanced_diagnostics"]):
        st.markdown(
            f"**{t['validation']['brier']}:** {brier:.3f} — {t['metric_help']['brier']}  \\n"
            f"**{t['validation']['ece']}:** {ece:.3f} — {t['metric_help']['calibration']}"
        )
        if calibration_candidates:
            calibration_rows = [
                {
                    "Method": method,
                    "Brier": metrics.get("brier_score"),
                    "ECE": metrics.get("expected_calibration_error"),
                    "Log loss": metrics.get("log_loss"),
                }
                for method, metrics in calibration_candidates.items()
            ]
            st.markdown(f"**{t['validation']['calibration_candidates']}**")
            st.dataframe(pd.DataFrame(calibration_rows).sort_values("Brier"), width="stretch", hide_index=True)


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

    _metric_cards(
        [
            {
                "label": str(t["operations"]["model"]),
                "value": _model_display_name(card.get("selected_model", info.get("model_name", "unknown"))),
                "help": "The model currently used by the API and dashboard." if lang == "en" else "Modelo utilizado actualmente por la API y el dashboard.",
            },
            {
                "label": str(t["operations"]["rows"]),
                "value": _fmt_int(info.get("n_train_rows")),
                "help": str(t["metric_help"]["training_rows"]),
            },
            {
                "label": str(t["operations"]["features"]),
                "value": str(len(info.get("feature_columns") or [])),
                "help": str(t["metric_help"]["features"]),
            },
        ],
        columns=3,
    )

    st.markdown(f"### {t['operations']['monitoring']}")
    if int(monitoring.get("total_predictions", 0) or 0) == 0:
        st.markdown(f'<div class="fr-note">{t["operations"]["no_traffic"]}</div>', unsafe_allow_html=True)
    drift_status = str(drift.get("status", "n/a")).upper()
    _metric_cards(
        [
            {
                "label": str(t["operations"]["predictions"]),
                "value": str(int(monitoring.get("total_predictions", 0) or 0)),
                "help": str(t["metric_help"]["predictions"]),
            },
            {
                "label": str(t["operations"]["average"]),
                "value": _fmt_pct(monitoring.get("average_probability")),
                "help": "Average predicted risk among recorded demo requests." if lang == "en" else "Riesgo predicho medio entre las peticiones registradas de la demo.",
            },
            {
                "label": str(t["operations"]["drift"]),
                "value": drift_status,
                "help": _drift_explanation(drift.get("status"), t),
            },
        ],
        columns=3,
    )

    with st.expander(t["operations"]["advanced_runtime"]):
        if drift.get("features"):
            drift_df = pd.DataFrame(
                [{"Feature": key, "PSI": value} for key, value in drift.get("features", {}).items()]
            ).sort_values("PSI", ascending=False)
            st.markdown(f"**{t['operations']['drift']}** — {t['metric_help']['drift']}")
            st.dataframe(drift_df, width="stretch", hide_index=True)

        st.markdown(f"**{t['operations']['performance']}**")
        performance_rows = pd.DataFrame(
            [
                {"Measurement": t["operations"]["artifact_load"], "Median": _fmt_ms(performance.get("artifact_load_ms"))},
                {"Measurement": t["operations"]["single_latency"], "Median": _fmt_ms(performance.get("single_prediction_ms", {}).get("median"))},
                {"Measurement": t["operations"]["batch_100"], "Median": _fmt_ms(performance.get("batch_100_ms", {}).get("median"))},
                {"Measurement": t["operations"]["batch_1000"], "Median": _fmt_ms(performance.get("batch_1000_ms", {}).get("median"))},
            ]
        )
        st.dataframe(performance_rows, width="stretch", hide_index=True)
        st.caption(t["operations"]["environment"])
        st.write({"Latest prediction UTC": monitoring.get("latest_prediction_utc") or "n/a"})

    with st.expander(t["operations"]["model_card"]):
        intended = card.get("intended_use", "Portfolio ML evaluation")
        not_intended = card.get("not_intended_use", "Operational aviation decisions")
        if lang == "es":
            intended = "Sistema educativo de portfolio para estimar riesgo de retraso con información del horario."
            not_intended = "Aviación operacional, seguridad, despacho o decisiones de viaje de alto impacto."
        st.markdown(
            f"""
- **Task / Tarea:** {card.get('task', 'Binary arrival-delay classification')}
- **Target:** `{card.get('target', 'ArrDel15')}` = arrival at least 15 minutes late / llegada con al menos 15 minutos de retraso
- **Release:** `v{APP_VERSION} · {RELEASE_NAME}`
- **Artifact:** `{info.get('version', 'unknown')}`
- **Calibration:** `{info.get('calibration_method', 'identity')}`
- **Held-out ranking quality (PR-AUC):** `{main_metrics.get('pr_auc', 'n/a')}`
- **Held-out probability error (Brier):** `{main_metrics.get('brier_score', 'n/a')}`
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
- agregados históricos construidos con fechas estrictamente anteriores

**Bloqueado explícitamente**

- retrasos y horas reales del propio vuelo
- taxi, wheels time, duración real y tiempo en el aire
- causas de retraso conocidas después de operar el vuelo
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
- historical aggregates built from strictly earlier dates

**Explicitly blocked**

- actual delays and actual times for the flight being scored
- taxi, wheels, actual elapsed and airborne times
- delay causes known after operating the flight
- cancellation and diversion status as inference features
"""
            )

    with st.expander(t["operations"]["api"]):
        st.code(
            """GET  /live
GET  /ready
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
    catalog = _safe_catalog()

    top_left, top_right = st.columns([0.84, 0.16], gap="small")
    with top_right:
        lang, t = _language_selector()
    with top_left:
        _topbar(model_available, info.get("version"), t)

    _hero(t, model_available, catalog)

    tabs = st.tabs(t["tabs"])
    with tabs[0]:
        _section_header(t["analyze_title"], t["analyze_sub"])
        form_result = _flight_form(lang, t, catalog, disabled=not model_available)
        if form_result is not None:
            payload, metadata = form_result
            _render_prediction(payload, metadata, lang, t)
        else:
            st.markdown(f'<div class="fr-note">{t["single"]["idle"]}</div>', unsafe_allow_html=True)
        _render_airport_heatmap(lang, t)

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
