"""FlightRisk product dashboard.

The interface is organised around four real product surfaces:
1. analyze one scheduled flight,
2. rank a schedule,
3. inspect validation evidence,
4. inspect model lineage and operating boundaries.
"""
from __future__ import annotations

import io
import json
import math
import sys
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.dashboard.i18n import TEXT
from app.dashboard.theme import CSS
from app.services import prediction_service
from src.models.predict import PredictionInput
from src.version import APP_VERSION, RELEASE_NAME

st.set_page_config(
    page_title="FlightRisk — Schedule Risk Workbench",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_flights.csv"


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
        "delay_probability": 0.238,
        "risk_level": "moderate",
        "decision_threshold": 0.52,
        "top_factors": ["sample schedule profile"],
    }
    fallback_context = {
        "route": "JFK → LAX",
        "global_rate": 0.227,
        "route_rate": 0.162,
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


def _hhmm(value: time) -> int:
    return value.hour * 100 + value.minute


def _risk_copy(level: str, lang: str) -> tuple[str, str]:
    if lang == "es":
        mapping = {
            "low": ("REVISIÓN RUTINARIA", "Perfil inferior a los niveles de mayor exposición del artefacto actual."),
            "moderate": ("VIGILAR", "El vuelo merece contexto adicional antes de tratar el score como señal fuerte."),
            "high": ("REVISIÓN PRIORITARIA", "El modelo sitúa este vuelo en una zona de exposición elevada."),
        }
    else:
        mapping = {
            "low": ("ROUTINE REVIEW", "The profile sits below the current artifact's higher-exposure range."),
            "moderate": ("WATCH", "The flight deserves additional context before treating the score as a strong signal."),
            "high": ("PRIORITY REVIEW", "The model places this flight in an elevated-exposure range."),
        }
    return mapping.get(level, mapping["moderate"])


def _topbar(model_available: bool, artifact_version: str | None) -> None:
    status_class = "ok" if model_available else ""
    status_text = "Model loaded" if model_available else "Model unavailable"
    artifact = artifact_version or "unknown"
    st.markdown(
        f"""
<div class="fr-topbar">
  <div class="fr-brand">
    <div class="fr-mark">FR</div>
    <div>
      <div class="fr-brand-name">FLIGHTRISK</div>
      <div class="fr-byline">Built by Oriol Martínez · ML product engineering</div>
    </div>
  </div>
  <div class="fr-status">
    <span class="fr-chip">v{APP_VERSION} · {RELEASE_NAME}</span>
    <span class="fr-chip">artifact {artifact}</span>
    <span class="fr-chip {status_class}">{status_text}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _hero(t: dict[str, Any], model_available: bool) -> None:
    prediction, context = _safe_sample_state(model_available)
    probability = float(prediction.get("delay_probability", 0.0))
    route_rate = float(context.get("route_rate", 0.0))
    global_rate = float(context.get("global_rate", 0.0))
    relative = probability / route_rate if route_rate > 0 else probability / max(global_rate, 1e-9)
    support = context.get("route_support_estimate")
    route = context.get("route", "JFK → LAX")
    priority = {"low": "ROUTINE", "moderate": "WATCH", "high": "PRIORITY"}.get(
        str(prediction.get("risk_level", "moderate")), "WATCH"
    )

    st.markdown('<div class="fr-hero">', unsafe_allow_html=True)
    left, right = st.columns([0.64, 0.36], gap="large")
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
      <div class="fr-flight-id">Example · DL 418 · 18:30 departure</div>
      <div class="fr-route">{route}</div>
    </div>
    <div class="fr-priority">{priority}</div>
  </div>
  <div class="fr-risk-number">{_fmt_pct(probability)}</div>
  <div class="fr-risk-label">current model probability · not post-calibrated</div>
  <div class="fr-flight-grid">
    <div class="fr-flight-stat"><b>{_fmt_pct(route_rate)}</b><span>route historical rate</span></div>
    <div class="fr-flight-stat"><b>{relative:.2f}×</b><span>score vs route cohort</span></div>
    <div class="fr-flight-stat"><b>{_fmt_int(support)}</b><span>estimated route support</span></div>
    <div class="fr-flight-stat"><b>{_fmt_pct(global_rate)}</b><span>training fallback rate</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="fr-section-head"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _flight_form(lang: str, catalog: dict[str, list[str]], disabled: bool) -> PredictionInput | None:
    carriers = catalog["carriers"]
    airports = catalog["airports"]
    default_carrier = carriers.index("DL") if "DL" in carriers else 0
    default_origin = airports.index("JFK") if "JFK" in airports else 0
    default_dest = airports.index("LAX") if "LAX" in airports else min(1, len(airports) - 1)

    labels = {
        "en": {
            "carrier": "Carrier",
            "origin": "Origin",
            "destination": "Destination",
            "date": "Flight date",
            "departure": "Scheduled departure",
            "arrival": "Scheduled arrival",
            "duration": "Scheduled duration (minutes)",
            "distance": "Distance (miles)",
            "submit": "Analyze flight",
        },
        "es": {
            "carrier": "Aerolínea",
            "origin": "Origen",
            "destination": "Destino",
            "date": "Fecha del vuelo",
            "departure": "Salida programada",
            "arrival": "Llegada programada",
            "duration": "Duración programada (minutos)",
            "distance": "Distancia (millas)",
            "submit": "Analizar vuelo",
        },
    }[lang]

    with st.form("single_flight_form"):
        c1, c2, c3 = st.columns(3)
        airline = c1.selectbox(labels["carrier"], carriers, index=default_carrier)
        origin = c2.selectbox(labels["origin"], airports, index=default_origin)
        destination = c3.selectbox(labels["destination"], airports, index=default_dest)

        c4, c5, c6 = st.columns(3)
        flight_date = c4.date_input(labels["date"], value=date.today())
        departure = c5.time_input(labels["departure"], value=time(18, 30), step=300)
        arrival = c6.time_input(labels["arrival"], value=time(21, 45), step=300)

        c7, c8 = st.columns(2)
        duration = c7.number_input(labels["duration"], min_value=20, max_value=900, value=375, step=5)
        distance = c8.number_input(labels["distance"], min_value=20.0, max_value=10000.0, value=2475.0, step=25.0)

        submitted = st.form_submit_button(labels["submit"], disabled=disabled, width="stretch")

    if not submitted:
        return None
    if origin == destination:
        st.error("Origin and destination must be different.")
        return None

    return PredictionInput(
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


def _render_context_rows(context: dict[str, Any]) -> None:
    signals = context.get("signals") or []
    for signal in signals[:4]:
        value = signal.get("value")
        baseline = signal.get("baseline")
        support = signal.get("support")
        direction = "above" if isinstance(value, (int, float)) and isinstance(baseline, (int, float)) and value >= baseline else "below"
        support_copy = f" · ~{_fmt_int(support)} rows" if support else ""
        st.markdown(
            f"""
<div class="fr-context-row">
  <div>
    <div class="fr-context-label">{signal.get('label', 'Historical context')}</div>
    <div class="fr-context-meta">{direction} global fallback{support_copy}</div>
  </div>
  <div class="fr-context-value">{_fmt_pct(value)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_prediction(payload: PredictionInput, lang: str) -> None:
    try:
        result = prediction_service.predict_flight(payload)
        context = prediction_service.prediction_context(payload)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    probability = float(result["delay_probability"])
    route_rate = float(context.get("route_rate", context.get("global_rate", 0.0)))
    global_rate = float(context.get("global_rate", 0.0))
    relative = probability / route_rate if route_rate > 0 else 0.0
    decision, explanation = _risk_copy(str(result.get("risk_level", "moderate")), lang)

    st.markdown(
        f"""
<div class="fr-decision">
  <div class="fr-decision-kicker">Schedule triage recommendation</div>
  <h3>{decision}</h3>
  <p>{explanation}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model probability", _fmt_pct(probability))
    m2.metric("Route cohort", _fmt_pct(route_rate))
    m3.metric("Relative exposure", f"{relative:.2f}×")
    m4.metric("Route support", _fmt_int(context.get("route_support_estimate")))

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        st.markdown('<div class="fr-panel"><b>Historical schedule context</b>', unsafe_allow_html=True)
        _render_context_rows(context)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        route_seen = "Seen in training" if context.get("route_seen") else "Unseen route fallback"
        carrier_route_seen = "Seen in training" if context.get("carrier_route_seen") else "Unseen carrier-route fallback"
        st.markdown(
            f"""
<div class="fr-panel">
  <b>Reliability context</b>
  <div class="fr-context-row"><div><div class="fr-context-label">Route coverage</div></div><div class="fr-context-value">{route_seen}</div></div>
  <div class="fr-context-row"><div><div class="fr-context-label">Carrier-route coverage</div></div><div class="fr-context-value">{carrier_route_seen}</div></div>
  <div class="fr-context-row"><div><div class="fr-context-label">Training fallback</div></div><div class="fr-context-value">{_fmt_pct(global_rate)}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="fr-note" style="margin-top:.65rem">The current committed artifact is useful mainly as a ranking signal. Its probability output has not yet been post-calibrated; calibration is explicitly shown in the Validation surface.</div>',
            unsafe_allow_html=True,
        )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower().replace(" ", "_") for column in normalized.columns]
    aliases = {
        "reporting_airline": "airline",
        "operating_airline": "airline",
        "dest": "destination",
        "origin_airport": "origin",
        "destination_airport": "destination",
        "crsdeptime": "crs_dep_time",
        "crsarrtime": "crs_arr_time",
        "crselapsedtime": "crs_elapsed_time",
        "dayofweek": "day_of_week",
    }
    return normalized.rename(columns=aliases)


def _payloads_from_df(df: pd.DataFrame) -> list[PredictionInput]:
    normalized = _normalize_columns(df)
    required = [
        "airline",
        "origin",
        "destination",
        "month",
        "day_of_week",
        "crs_dep_time",
        "crs_arr_time",
        "crs_elapsed_time",
        "distance",
    ]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    payloads: list[PredictionInput] = []
    for _, row in normalized[required].iterrows():
        payloads.append(
            PredictionInput(
                airline=str(row["airline"]).strip().upper(),
                origin=str(row["origin"]).strip().upper(),
                destination=str(row["destination"]).strip().upper(),
                month=int(row["month"]),
                day_of_week=int(row["day_of_week"]),
                crs_dep_time=int(row["crs_dep_time"]),
                crs_arr_time=int(row["crs_arr_time"]),
                crs_elapsed_time=int(row["crs_elapsed_time"]),
                distance=float(row["distance"]),
            )
        )
    return payloads


def _sample_batch() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        sample = pd.read_csv(SAMPLE_PATH).head(18)
        rename_map = {
            "Airline": "airline",
            "Origin": "origin",
            "Dest": "destination",
            "Month": "month",
            "DayOfWeek": "day_of_week",
            "CRSDepTime": "crs_dep_time",
            "CRSArrTime": "crs_arr_time",
            "CRSElapsedTime": "crs_elapsed_time",
            "Distance": "distance",
        }
        available = {key: value for key, value in rename_map.items() if key in sample.columns}
        sample = sample.rename(columns=available)
        required = list(rename_map.values())
        if all(column in sample.columns for column in required):
            return sample[required]

    return pd.DataFrame(
        [
            ["DL", "JFK", "LAX", 7, 5, 1830, 2145, 375, 2475],
            ["UA", "SFO", "EWR", 7, 5, 2215, 630, 315, 2565],
            ["AA", "ORD", "DFW", 7, 5, 1720, 1955, 155, 802],
            ["WN", "ATL", "BOS", 7, 5, 805, 1035, 150, 946],
        ],
        columns=[
            "airline", "origin", "destination", "month", "day_of_week",
            "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
        ],
    )


def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    payloads = _payloads_from_df(df)
    normalized = _normalize_columns(df).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_service.predict_flights_batch(payloads))
    ranked = pd.concat([normalized, predictions], axis=1)
    ranked = ranked.sort_values("delay_probability", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["delay_probability_pct"] = (ranked["delay_probability"] * 100).round(1)
    priority_count = max(1, math.ceil(len(ranked) * 0.10))
    watch_cutoff = max(priority_count, math.ceil(len(ranked) * 0.30))
    ranked["priority_tier"] = "Routine"
    ranked.loc[ranked["rank"] <= watch_cutoff, "priority_tier"] = "Watch"
    ranked.loc[ranked["rank"] <= priority_count, "priority_tier"] = "Priority"
    return ranked


def _render_batch(model_available: bool) -> None:
    st.markdown(
        '<div class="fr-note">Required columns: airline, origin, destination, month, day_of_week, crs_dep_time, crs_arr_time, crs_elapsed_time, distance.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.72, 0.28], gap="large")
    with left:
        uploaded = st.file_uploader("Upload schedule CSV", type=["csv"], disabled=not model_available)
    with right:
        st.write("")
        use_sample = st.button("Load sample schedule", disabled=not model_available, width="stretch")

    dataframe: pd.DataFrame | None = None
    if uploaded is not None:
        dataframe = pd.read_csv(uploaded)
    elif use_sample:
        dataframe = _sample_batch()

    if dataframe is None or not model_available:
        return

    try:
        ranked = rank_dataframe(dataframe)
    except Exception as exc:
        st.error(str(exc))
        return

    total = len(ranked)
    priority = int((ranked["priority_tier"] == "Priority").sum())
    watch = int((ranked["priority_tier"] == "Watch").sum())
    avg = float(ranked["delay_probability"].mean()) if total else 0.0
    max_probability = float(ranked["delay_probability"].max()) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flights ranked", f"{total}")
    m2.metric("Priority queue", f"{priority}")
    m3.metric("Watch queue", f"{watch}")
    m4.metric("Highest score", _fmt_pct(max_probability))

    display_columns = [
        "rank", "airline", "origin", "destination", "crs_dep_time",
        "delay_probability_pct", "priority_tier", "risk_level",
    ]
    existing = [column for column in display_columns if column in ranked.columns]
    st.dataframe(
        ranked[existing],
        width="stretch",
        hide_index=True,
        column_config={
            "delay_probability_pct": st.column_config.NumberColumn("Model probability", format="%.1f%%"),
            "priority_tier": st.column_config.TextColumn("Review queue"),
            "crs_dep_time": st.column_config.NumberColumn("Departure", format="%04d"),
        },
    )

    distribution = ranked["priority_tier"].value_counts().reindex(["Priority", "Watch", "Routine"]).fillna(0)
    st.bar_chart(distribution, height=230)

    csv_buffer = io.StringIO()
    ranked.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download ranked schedule",
        data=csv_buffer.getvalue(),
        file_name="flightrisk_ranked_schedule.csv",
        mime="text/csv",
    )
    st.caption(f"Average model probability across this schedule: {_fmt_pct(avg)}.")


def _model_metrics(report: dict[str, Any], key: str) -> dict[str, Any]:
    block = report.get(key, {}) if isinstance(report, dict) else {}
    return block.get("metrics", block) if isinstance(block, dict) else {}


def _render_validation() -> None:
    report = _load_json(REPORTS_DIR / "metrics.json")
    main_metrics = _model_metrics(report, "main_model")
    baseline_metrics = _model_metrics(report, "baseline_model")
    selected_name = report.get("main_model", {}).get("model_name", "random_forest")
    baseline_name = report.get("baseline_model", {}).get("model_name", "logistic_regression")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Held-out PR-AUC", f"{main_metrics.get('pr_auc', 0):.3f}")
    m2.metric("Lift@10%", f"{main_metrics.get('lift_at_top_10pct', 0):.2f}×")
    m3.metric("Top-10% precision", _fmt_pct(main_metrics.get("precision_at_top_10pct")))
    m4.metric("Held-out rows", _fmt_int(main_metrics.get("n_samples")))

    st.markdown(
        '<div class="fr-note">Honest result: the Logistic Regression baseline generalized slightly better than the validation-selected Random Forest on the final held-out period. This is retained as evidence of model-selection instability, not hidden.</div>',
        unsafe_allow_html=True,
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": selected_name,
                "ROC-AUC": main_metrics.get("roc_auc"),
                "PR-AUC": main_metrics.get("pr_auc"),
                "F1": main_metrics.get("f1"),
                "Precision@10%": main_metrics.get("precision_at_top_10pct"),
                "Lift@10%": main_metrics.get("lift_at_top_10pct"),
            },
            {
                "Model": baseline_name,
                "ROC-AUC": baseline_metrics.get("roc_auc"),
                "PR-AUC": baseline_metrics.get("pr_auc"),
                "F1": baseline_metrics.get("f1"),
                "Precision@10%": baseline_metrics.get("precision_at_top_10pct"),
                "Lift@10%": baseline_metrics.get("lift_at_top_10pct"),
            },
        ]
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

    left, right = st.columns([0.56, 0.44], gap="large")
    with left:
        calibration = report.get("main_model", {}).get("calibration", {})
        predicted = calibration.get("mean_predicted_probability") or []
        observed = calibration.get("fraction_of_positives") or []
        if predicted and observed and len(predicted) == len(observed):
            calibration_df = pd.DataFrame(
                {"Observed frequency": observed, "Perfect calibration": predicted},
                index=pd.Index(predicted, name="Mean predicted probability"),
            )
            st.markdown("**Calibration diagnostic**")
            st.line_chart(calibration_df, height=300)
            st.caption("The gap between observed frequency and the diagonal is why the current artifact should be treated mainly as a ranking signal.")
    with right:
        st.markdown(
            """
<div class="fr-validation-card"><b>Temporal holdout</b><span>Earlier flights train the model; later flights form validation and test periods. v0.8 also prevents identical FlightDate values from crossing a split boundary.</span></div>
<br>
<div class="fr-validation-card"><b>Leakage contract</b><span>Actual delays, taxi times, wheels times, actual elapsed time and delay causes are explicitly blocked from model features.</span></div>
<br>
<div class="fr-validation-card"><b>Next experimental gate</b><span>Ordered historical encoding, temporal backtesting and post-hoc calibration remain the next major model iteration.</span></div>
""",
            unsafe_allow_html=True,
        )

    selection = report.get("selection", {}).get("validation_metrics", {})
    if selection:
        rows = []
        for model_name, metrics in selection.items():
            rows.append(
                {
                    "Candidate": model_name,
                    "Validation PR-AUC": metrics.get("pr_auc"),
                    "Validation ROC-AUC": metrics.get("roc_auc"),
                    "Lift@10%": metrics.get("lift_at_top_10pct"),
                }
            )
        st.markdown("**Validation candidate comparison**")
        st.dataframe(pd.DataFrame(rows).sort_values("Validation PR-AUC", ascending=False), width="stretch", hide_index=True)


def _render_operations(info: dict[str, Any], card: dict[str, Any]) -> None:
    metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
    main = metrics.get("main_model", {}) if isinstance(metrics, dict) else {}
    main_metrics = main.get("metrics", main) if isinstance(main, dict) else {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
<div class="fr-validation-card"><b>{card.get('selected_model', info.get('model_name', 'unknown'))}</b><span>Current serialized classifier selected on validation PR-AUC.</span></div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
<div class="fr-validation-card"><b>{_fmt_int(info.get('n_train_rows'))} training rows</b><span>Real BTS Reporting Carrier On-Time Performance data.</span></div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
<div class="fr-validation-card"><b>{len(info.get('feature_columns') or [])} model features</b><span>Schedule, calendar, route and train-fitted historical aggregates.</span></div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("Model card", expanded=True):
        st.markdown(
            f"""
- **Task:** {card.get('task', 'Binary arrival-delay classification')}
- **Target:** `{card.get('target', 'ArrDel15')}`
- **Product release:** `v{APP_VERSION}`
- **Serialized artifact metadata:** `{info.get('version', 'unknown')}`
- **Decision threshold:** `{info.get('decision_threshold', 'n/a')}`
- **Held-out PR-AUC:** `{main_metrics.get('pr_auc', 'n/a')}`
- **Intended use:** {card.get('intended_use', 'Portfolio ML evaluation')}
- **Not intended for:** {card.get('not_intended_use', 'Operational aviation decisions')}
"""
        )

    with st.expander("Pre-departure leakage contract"):
        st.markdown(
            """
**Allowed before departure**

- carrier, origin and destination
- calendar and scheduled times
- scheduled duration and distance
- historical aggregates fitted from training data

**Explicitly blocked**

- `ArrDelay`, `DepDelay`, `ArrDelayMinutes`
- actual departure/arrival, taxi and wheels times
- actual elapsed time and airborne time
- carrier, weather, NAS and late-aircraft delay causes
- cancellation and diversion status as inference features
"""
        )

    with st.expander("API surface"):
        st.code(
            """GET  /health
GET  /model/info
GET  /model/card
POST /predict
POST /predict/batch
POST /rank
GET  /monitoring/summary
GET  /monitoring/drift""",
            language="text",
        )
        st.markdown("FastAPI provides interactive OpenAPI documentation at `/docs`.")

    with st.expander("Repository architecture"):
        st.code(
            """app/api/           FastAPI transport layer
app/dashboard/     Streamlit product surface
app/services/      inference service layer
src/data/          loading, cleaning, temporal splitting
src/features/      schedule features and historical aggregates
src/models/        training, evaluation, registry, inference
src/monitoring/    prediction logging and PSI drift checks
scripts/           reproducible CLI workflows
reports/           committed evaluation evidence""",
            language="text",
        )


def main() -> None:
    _inject_theme()
    info = _safe_model_info()
    card = _safe_model_card()
    model_available = prediction_service.is_model_available()

    top_left, top_right = st.columns([0.84, 0.16], gap="small")
    with top_left:
        _topbar(model_available, info.get("version"))
    with top_right:
        lang, t = _language_selector()

    _hero(t, model_available)

    tabs = st.tabs(t["tabs"])
    with tabs[0]:
        _section_header(t["analyze_title"], t["analyze_sub"])
        payload = _flight_form(lang, _safe_catalog(), disabled=not model_available)
        if payload is not None:
            _render_prediction(payload, lang)
        else:
            st.markdown(
                '<div class="fr-note">The form converts date and clock inputs into the exact schedule features required by the model. No post-departure field is requested.</div>',
                unsafe_allow_html=True,
            )

    with tabs[1]:
        _section_header(t["rank_title"], t["rank_sub"])
        _render_batch(model_available)

    with tabs[2]:
        _section_header(t["validation_title"], t["validation_sub"])
        _render_validation()

    with tabs[3]:
        _section_header(t["operations_title"], t["operations_sub"])
        _render_operations(info, card)

    st.markdown(
        '<div class="fr-footer">FlightRisk · Built by Oriol Martínez · Portfolio ML system · Not for operational aviation decisions.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
