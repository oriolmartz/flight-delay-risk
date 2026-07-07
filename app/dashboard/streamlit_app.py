"""FlightRisk simple Streamlit UI.

English is the default language. The page intentionally explains one idea first:
FlightRisk estimates the probability that a scheduled flight arrives 15+ minutes late.
Batch ranking and Europe context are secondary workflows.
"""
from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.services import prediction_service
from src.models.predict import PredictionInput

st.set_page_config(
    page_title="FlightRisk — Delay Probability Model",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root {
  --ink-950: #07162b;
  --ink-900: #0d2845;
  --ink-800: #173d64;
  --slate-700: #355d7c;
  --blue-500: #2f8fd8;
  --sky-300: #74d2ee;
  --sky-100: #eaf7ff;
  --paper: #f3f8fd;
  --paper-2: #e7f0f8;
  --muted: #627a93;
  --text: #10283f;
  --panel: rgba(255,255,255,.78);
  --panel-blue: rgba(14, 48, 80, .78);
  --line: rgba(47,143,216,.18);
  --line-dark: rgba(7,22,43,.12);
  --green: #32b37b;
  --amber: #e5a548;
  --red: #e86373;
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  color: var(--text);
  background:
    radial-gradient(circle at 18% 0%, rgba(116,210,238,.34), transparent 25%),
    radial-gradient(circle at 84% 8%, rgba(47,143,216,.18), transparent 28%),
    linear-gradient(180deg, #dcebf6 0%, #eef5fb 42%, #f7fbff 100%);
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.main .block-container { max-width: 1240px; padding-top: .9rem; padding-bottom: 3rem; }
h1, h2, h3, h4, p, div, span, label { color: var(--text); }
.stSelectbox label, .stFileUploader label, .stTextInput label, .stNumberInput label { color: var(--text) !important; font-weight: 800; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: rgba(255,255,255,.86);
  border: 1px solid var(--line-dark);
  color: var(--text);
}
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
  background: linear-gradient(180deg, #37a8e8, #2477be) !important;
  color: #ffffff !important;
  border: 1px solid rgba(47,143,216,.25) !important;
  border-radius: 11px !important;
  font-weight: 850 !important;
  box-shadow: 0 8px 18px rgba(32,105,164,.16);
}
[data-testid="stMetric"] {
  background: rgba(255,255,255,.82);
  border: 1px solid var(--line-dark);
  border-radius: 16px;
  padding: .75rem;
  box-shadow: 0 8px 22px rgba(22,55,84,.07);
}
[data-testid="stMetric"] label, [data-testid="stMetric"] div { color: var(--text) !important; }
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; border:1px solid var(--line-dark); }
.fr-topbar {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:.9rem 1rem; background:rgba(255,255,255,.72); border:1px solid var(--line-dark);
  border-radius:18px; box-shadow:0 14px 32px rgba(19,55,85,.10); backdrop-filter: blur(12px);
}
.fr-logo { display:flex; align-items:center; gap:.65rem; font-size:1.08rem; font-weight:950; letter-spacing:.13em; color:var(--ink-900); }
.fr-logo-mark { color:var(--blue-500); font-size:1.36rem; }
.fr-nav { display:flex; flex-wrap:wrap; gap:1rem; align-items:center; color:#355d7c; font-size:.86rem; font-weight:750; }
.fr-nav span { color:#355d7c; }
.fr-hero {
  position:relative; overflow:hidden; margin:.85rem 0 1rem 0; padding:2rem; border-radius:28px;
  background:
    linear-gradient(rgba(47,143,216,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(47,143,216,.045) 1px, transparent 1px),
    radial-gradient(circle at 80% 15%, rgba(116,210,238,.28), transparent 25%),
    linear-gradient(112deg, #f8fbff 0%, #e6f2fb 52%, #cfe4f4 100%);
  background-size: 48px 48px, 48px 48px, auto, auto;
  box-shadow:0 26px 62px rgba(19,55,85,.16); border:1px solid rgba(47,143,216,.18);
}
.fr-hero-grid { position:relative; z-index:2; display:grid; grid-template-columns:minmax(360px,1fr) minmax(410px,.92fr); gap:1.45rem; align-items:center; }
.fr-tower {
  position:absolute; right:4%; bottom:0; width:210px; height:310px; z-index:1; opacity:.15; pointer-events:none;
  filter: drop-shadow(0 0 18px rgba(47,143,216,.12));
}
.fr-tower .cab { position:absolute; left:42px; top:58px; width:130px; height:48px; border-radius:16px 16px 8px 8px; background:linear-gradient(180deg, rgba(74,140,190,.34), rgba(255,255,255,.38)); border:1px solid rgba(47,143,216,.35); }
.fr-tower .cab:before { content:""; position:absolute; left:-20px; top:14px; width:170px; height:12px; border-radius:999px; background:rgba(47,143,216,.22); }
.fr-tower .stem { position:absolute; left:88px; top:106px; width:40px; height:205px; background:linear-gradient(180deg, rgba(58,115,171,.28), rgba(255,255,255,.30)); clip-path: polygon(15% 0, 85% 0, 100% 100%, 0 100%); }
.fr-tower .antenna { position:absolute; left:106px; top:22px; width:4px; height:43px; background:rgba(47,143,216,.35); }
.fr-plane-line { display:none; }
.fr-aircraft-card {
  position:relative; overflow:hidden; min-height:150px; border-radius:20px; margin-bottom:.9rem;
  background:
    linear-gradient(180deg, rgba(255,255,255,.72), rgba(234,246,255,.70)),
    radial-gradient(circle at 20% 20%, rgba(116,210,238,.28), transparent 28%);
  border:1px solid rgba(47,143,216,.17);
}
.fr-aircraft-card:before {
  content:""; position:absolute; left:-8%; right:-8%; bottom:29px; height:42px;
  background:linear-gradient(90deg, transparent, rgba(15,54,88,.12), transparent); transform:rotate(-2deg);
}
.fr-aircraft {
  position:absolute; right:9%; top:25%; font-size:4.6rem; color:#2f8fd8; transform:rotate(-8deg);
  text-shadow: 0 12px 28px rgba(22,89,146,.16); opacity:.92;
}
.fr-aircraft-copy { position:absolute; left:1rem; bottom:.85rem; right:8rem; color:#355d7c; font-size:.88rem; line-height:1.45; }
.fr-aircraft-copy b { color:#10283f; }
.fr-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.4rem .72rem; border-radius:999px; background:rgba(47,143,216,.10); border:1px solid rgba(47,143,216,.20); color:#176599; font-size:.76rem; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
.fr-title { margin:.8rem 0 .6rem 0; font-size:clamp(2.05rem,3.7vw,3.95rem); line-height:1.02; font-weight:950; letter-spacing:-.052em; color:#10283f; max-width:780px; }
.fr-title span { color:#1986c8; }
.fr-subtitle { max-width:760px; font-size:1.02rem; line-height:1.65; color:#3f5e78; }
.fr-card {
  background:rgba(255,255,255,.80);
  border:1px solid var(--line-dark); border-radius:22px; padding:1.1rem; box-shadow:0 18px 42px rgba(20,65,101,.12);
}
.fr-card h3 { margin:.15rem 0 .6rem 0; font-size:1rem; letter-spacing:.02em; color:#10283f; }
.fr-mini { color:#627a93; font-size:.9rem; line-height:1.5; }
.fr-output { display:grid; grid-template-columns: repeat(3, 1fr); gap:.7rem; margin-top:.8rem; }
.fr-output-box { padding:.85rem; border-radius:16px; background:rgba(239,247,252,.92); border:1px solid rgba(47,143,216,.14); }
.fr-output-box .k { color:#627a93; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:900; }
.fr-output-box .v { color:#10283f; font-size:1.45rem; font-weight:950; margin-top:.2rem; }
.fr-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.85rem; margin:1rem 0; }
.fr-step { background:rgba(255,255,255,.82); border:1px solid var(--line-dark); border-radius:18px; padding:1rem; box-shadow:0 10px 28px rgba(20,65,101,.06); }
.fr-step b { color:#10283f; }
.fr-muted { color:#627a93; }
.fr-section-title { margin:1.3rem 0 .65rem 0; font-size:1.35rem; font-weight:950; letter-spacing:-.02em; color:#10283f; }
.fr-truth { padding:1rem; border-radius:18px; border:1px solid var(--line-dark); background:rgba(255,255,255,.58); }
.fr-warning { padding:.85rem 1rem; border-radius:16px; border:1px solid rgba(229,165,72,.25); background:rgba(229,165,72,.10); color:#5d4114; }
.fr-footer { margin-top:1.2rem; color:#627a93; font-size:.86rem; text-align:center; }
@media (max-width: 900px) {
  .fr-hero-grid, .fr-strip { grid-template-columns: 1fr; }
  .fr-output { grid-template-columns: 1fr; }
  .fr-aircraft-card { min-height: 120px; }
  .fr-aircraft { font-size:3.5rem; }
}
</style>
"""

TEXT = {
    "en": {
        "lang_label": "Language",
        "nav": ["Predict", "Batch", "Model", "Europe"],
        "status": "Model loaded",
        "hero_pill": "Flight delay probability model",
        "hero_title": "Will this flight arrive <span>15+ minutes late?</span>",
        "hero_sub": "FlightRisk estimates the probability that a scheduled flight arrives 15+ minutes late before departure. Then, in batch mode, it sorts flights from highest to lowest predicted delay probability.",
        "truth_title": "What the model predicts",
        "truth_body": "Target: P(ArrDel15 = 1). ArrDel15 means the arrival delay is 15 minutes or more.",
        "input_title": "Predict one scheduled flight",
        "input_help": "Only pre-flight schedule fields are used. No actual delay, taxi, wheels, cancellation or post-flight data.",
        "airline": "Airline",
        "origin": "Origin",
        "dest": "Destination",
        "month": "Month",
        "dow": "Day of week",
        "dep": "Scheduled departure time",
        "arr": "Scheduled arrival time",
        "elapsed": "Scheduled duration",
        "distance": "Distance",
        "predict": "Estimate delay probability",
        "prob": "Delay probability",
        "bucket": "Risk level",
        "target": "Target",
        "drivers": "Main signals",
        "steps": [
            ("1. Enter scheduled flight data", "Airline, route, date, scheduled times, duration and distance."),
            ("2. Model estimates delay probability", "The output is P(ArrDel15 = 1), not a guarantee."),
            ("3. Batch mode sorts by risk", "For many flights, highest probability appears first."),
        ],
        "batch_title": "Batch mode",
        "batch_body": "Upload a CSV with scheduled flights. FlightRisk adds delay probability, risk level and rank.",
        "csv_cols": "Required columns: airline, origin, destination, month, day_of_week, crs_dep_time, crs_arr_time, crs_elapsed_time, distance.",
        "sample": "Use sample batch",
        "upload": "Upload flight CSV",
        "download": "Download ranked CSV",
        "model_title": "Model details",
        "main_model": "Main model",
        "main_model_body": "Trained on BTS U.S. On-Time Performance flight-level data.",
        "target_body": "Target variable: ArrDel15 = arrival delay of 15+ minutes.",
        "features_body": "Inputs: airline, origin, destination, month, day of week, scheduled times, duration, distance and historical aggregate rates fitted from training data.",
        "metrics_title": "Bundled evaluation snapshot",
        "europe_title": "Europe context",
        "europe_body": "The Europe layer uses UK CAA aggregate punctuality data as route/airline/month context. It is experimental and is not the core flight-level training set.",
        "footer": "Portfolio ML project. Not for safety-critical dispatch, legal, compensation or operational aviation decisions.",
    },
    "es": {
        "lang_label": "Idioma",
        "nav": ["Predecir", "Batch", "Modelo", "Europa"],
        "status": "Modelo cargado",
        "hero_pill": "Modelo de probabilidad de retraso",
        "hero_title": "¿Llegará este vuelo con <span>15+ minutos de retraso?</span>",
        "hero_sub": "FlightRisk estima antes de la salida la probabilidad de que un vuelo programado llegue con 15 minutos o más de retraso. En modo batch, ordena los vuelos de mayor a menor probabilidad estimada de retraso.",
        "truth_title": "Qué predice el modelo",
        "truth_body": "Target: P(ArrDel15 = 1). ArrDel15 significa que el retraso de llegada es de 15 minutos o más.",
        "input_title": "Predecir un vuelo programado",
        "input_help": "Solo usa campos conocidos antes del vuelo. No usa retraso real, taxi, wheels, cancelación ni datos posteriores al vuelo.",
        "airline": "Aerolínea",
        "origin": "Origen",
        "dest": "Destino",
        "month": "Mes",
        "dow": "Día de la semana",
        "dep": "Hora salida programada",
        "arr": "Hora llegada programada",
        "elapsed": "Duración programada",
        "distance": "Distancia",
        "predict": "Estimar probabilidad de retraso",
        "prob": "Probabilidad de retraso",
        "bucket": "Nivel de riesgo",
        "target": "Target",
        "drivers": "Señales principales",
        "steps": [
            ("1. Introduce datos programados", "Aerolínea, ruta, fecha, horas programadas, duración y distancia."),
            ("2. El modelo estima probabilidad", "La salida es P(ArrDel15 = 1), no una garantía."),
            ("3. El batch ordena por riesgo", "Para muchos vuelos, aparece primero la mayor probabilidad."),
        ],
        "batch_title": "Modo batch",
        "batch_body": "Sube un CSV con vuelos programados. FlightRisk añade probabilidad de retraso, nivel de riesgo y ranking.",
        "csv_cols": "Columnas requeridas: airline, origin, destination, month, day_of_week, crs_dep_time, crs_arr_time, crs_elapsed_time, distance.",
        "sample": "Usar ejemplo batch",
        "upload": "Subir CSV de vuelos",
        "download": "Descargar CSV ordenado",
        "model_title": "Detalles del modelo",
        "main_model": "Modelo principal",
        "main_model_body": "Entrenado con datos BTS U.S. On-Time Performance a nivel vuelo individual.",
        "target_body": "Variable objetivo: ArrDel15 = retraso de llegada de 15+ minutos.",
        "features_body": "Inputs: aerolínea, origen, destino, mes, día de semana, horas programadas, duración, distancia y tasas históricas agregadas ajustadas en training.",
        "metrics_title": "Snapshot de evaluación incluido",
        "europe_title": "Contexto Europa",
        "europe_body": "La capa Europa usa datos agregados UK CAA de puntualidad como contexto de ruta/aerolínea/mes. Es experimental y no es el conjunto principal de entrenamiento flight-level.",
        "footer": "Proyecto portfolio ML. No usar para dispatch crítico, decisiones legales, compensación o aviación operacional.",
    },
}


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _language_selector() -> tuple[str, dict[str, Any]]:
    left, right = st.columns([0.82, 0.18])
    with right:
        choice = st.selectbox("Language / Idioma", ["English", "Español"], index=0, key="language_selector")
    lang = "en" if choice == "English" else "es"
    return lang, TEXT[lang]


def _safe_model_info() -> dict[str, Any]:
    try:
        return prediction_service.model_info()
    except Exception:
        return {}


def _topbar(t: dict[str, Any], model_available: bool) -> None:
    status = t["status"] if model_available else "Model not available"
    nav = "".join(f"<span>{item}</span>" for item in t["nav"])
    st.markdown(
        f"""
<div class="fr-topbar">
  <div class="fr-logo"><span class="fr-logo-mark">✈</span><span>FLIGHTRISK</span></div>
  <div class="fr-nav">{nav}<span>•</span><span>{status}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _tower_svg() -> str:
    return """
<div class="fr-tower" aria-hidden="true">
  <div class="antenna"></div>
  <div class="cab"></div>
  <div class="stem"></div>
</div>
"""


def _aircraft_visual(lang: str) -> str:
    copy = (
        "A schedule-time model for estimating arrival delay probability before departure."
        if lang == "en"
        else "Un modelo de horarios programados para estimar retraso antes de la salida."
    )
    title = "Pre-departure delay risk" if lang == "en" else "Riesgo antes de salida"
    return f"""
<div class="fr-aircraft-card" aria-hidden="true">
  <div class="fr-aircraft">✈</div>
  <div class="fr-aircraft-copy"><b>{title}</b><br>{copy}</div>
</div>
"""


def _prediction_payload_from_form(prefix: str = "single") -> PredictionInput:
    c1, c2, c3 = st.columns(3)
    with c1:
        airline = st.text_input("Airline", "DL", key=f"{prefix}_airline").upper()
        month = st.number_input("Month", min_value=1, max_value=12, value=7, step=1, key=f"{prefix}_month")
        dep = st.number_input("Scheduled departure time", min_value=0, max_value=2400, value=1830, step=5, key=f"{prefix}_dep")
    with c2:
        origin = st.text_input("Origin", "JFK", key=f"{prefix}_origin").upper()
        dow = st.number_input("Day of week", min_value=1, max_value=7, value=5, step=1, key=f"{prefix}_dow")
        arr = st.number_input("Scheduled arrival time", min_value=0, max_value=2400, value=2145, step=5, key=f"{prefix}_arr")
    with c3:
        dest = st.text_input("Destination", "LAX", key=f"{prefix}_dest").upper()
        elapsed = st.number_input("Scheduled duration", min_value=1, value=375, step=5, key=f"{prefix}_elapsed")
        distance = st.number_input("Distance", min_value=1.0, value=2475.0, step=10.0, key=f"{prefix}_distance")
    return PredictionInput(
        airline=airline,
        origin=origin,
        destination=dest,
        month=int(month),
        day_of_week=int(dow),
        crs_dep_time=int(dep),
        crs_arr_time=int(arr),
        crs_elapsed_time=int(elapsed),
        distance=float(distance),
    )


def _risk_label(level: str, lang: str) -> str:
    mapping_en = {"low": "Low", "moderate": "Moderate", "high": "High"}
    mapping_es = {"low": "Bajo", "moderate": "Moderado", "high": "Alto"}
    return (mapping_en if lang == "en" else mapping_es).get(level, level)


def _hero(t: dict[str, Any], lang: str, model_available: bool) -> None:
    sample_input = PredictionInput(
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
    sample_result: dict[str, Any] = {"delay_probability": 0.238, "risk_level": "moderate", "top_factors": ["sample route profile"]}
    if model_available:
        try:
            sample_result = prediction_service.predict_flight(sample_input)
        except Exception:
            pass

    prob = float(sample_result.get("delay_probability", 0.238)) * 100
    risk = _risk_label(str(sample_result.get("risk_level", "moderate")), lang)

    output_title = "Example output" if lang == "en" else "Ejemplo de salida"
    probability_note = (
        "P(ArrDel15 = 1) · probability of arriving 15+ minutes late"
        if lang == "en"
        else "P(ArrDel15 = 1) · probabilidad de llegar 15+ minutos tarde"
    )

    st.markdown(
        f"""
<section class="fr-hero">
  {_tower_svg()}
  <div class="fr-hero-grid">
    <div>
      <div class="fr-pill">{t['hero_pill']}</div>
      <div class="fr-title">{t['hero_title']}</div>
      <div class="fr-subtitle">{t['hero_sub']}</div>
      <div class="fr-truth" style="margin-top:1rem;">
        <b>{t['truth_title']}</b><br>
        <span class="fr-muted">{t['truth_body']}</span>
      </div>
    </div>
    <div class="fr-card">
      {_aircraft_visual(lang)}
      <h3>{output_title}</h3>
      <div class="fr-mini">DL · JFK → LAX · 18:30 · 375 min · 2,475 mi</div>
      <div class="fr-output">
        <div class="fr-output-box"><div class="k">{t['prob']}</div><div class="v">{prob:.1f}%</div></div>
        <div class="fr-output-box"><div class="k">{t['bucket']}</div><div class="v">{risk}</div></div>
        <div class="fr-output-box"><div class="k">{t['target']}</div><div class="v">ArrDel15</div></div>
      </div>
      <div class="fr-mini" style="margin-top:.9rem;">{probability_note}</div>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def _steps(t: dict[str, Any]) -> None:
    html = '<div class="fr-strip">'
    for title, body in t["steps"]:
        html += f'<div class="fr-step"><b>{title}</b><br><span class="fr-muted">{body}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _single_prediction(t: dict[str, Any], lang: str, model_available: bool) -> None:
    st.markdown(f'<div class="fr-section-title">{t["input_title"]}</div>', unsafe_allow_html=True)
    st.caption(t["input_help"])
    with st.form("single_prediction_form"):
        payload = _prediction_payload_from_form("single")
        submitted = st.form_submit_button(t["predict"], disabled=not model_available)
    if submitted and model_available:
        result = prediction_service.predict_flight(payload)
        prob = float(result.get("delay_probability", 0.0)) * 100
        risk = _risk_label(str(result.get("risk_level", "")), lang)
        c1, c2, c3 = st.columns(3)
        c1.metric(t["prob"], f"{prob:.1f}%")
        c2.metric(t["bucket"], risk)
        c3.metric(t["target"], "ArrDel15")
        factors = result.get("top_factors", [])
        if factors:
            st.write(f"**{t['drivers']}**")
            for factor in factors:
                st.write(f"• {factor}")


def _sample_batch() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"airline": "DL", "origin": "JFK", "destination": "LAX", "month": 7, "day_of_week": 5, "crs_dep_time": 1830, "crs_arr_time": 2145, "crs_elapsed_time": 375, "distance": 2475},
            {"airline": "AA", "origin": "ORD", "destination": "DFW", "month": 7, "day_of_week": 1, "crs_dep_time": 1720, "crs_arr_time": 1955, "crs_elapsed_time": 155, "distance": 802},
            {"airline": "UA", "origin": "SFO", "destination": "EWR", "month": 8, "day_of_week": 4, "crs_dep_time": 2215, "crs_arr_time": 640, "crs_elapsed_time": 325, "distance": 2565},
            {"airline": "WN", "origin": "LAS", "destination": "DEN", "month": 6, "day_of_week": 7, "crs_dep_time": 930, "crs_arr_time": 1220, "crs_elapsed_time": 110, "distance": 628},
            {"airline": "B6", "origin": "BOS", "destination": "MCO", "month": 12, "day_of_week": 6, "crs_dep_time": 1545, "crs_arr_time": 1910, "crs_elapsed_time": 205, "distance": 1121},
        ]
    )


def _payloads_from_df(df: pd.DataFrame) -> list[PredictionInput]:
    rename = {
        "Airline": "airline",
        "Origin": "origin",
        "Dest": "destination",
        "Destination": "destination",
        "Month": "month",
        "DayOfWeek": "day_of_week",
        "DayofWeek": "day_of_week",
        "CRSDepTime": "crs_dep_time",
        "CRSArrTime": "crs_arr_time",
        "CRSElapsedTime": "crs_elapsed_time",
        "Distance": "distance",
    }
    normalized = df.rename(columns=rename).copy()
    required = ["airline", "origin", "destination", "month", "day_of_week", "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance"]
    missing = [col for col in required if col not in normalized.columns]
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


def rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe sorted by predicted delay probability."""
    payloads = _payloads_from_df(df)
    preds = prediction_service.rank_flights_batch(payloads).get("ranked_predictions", [])

    original = df.reset_index(drop=False).rename(columns={"index": "_original_index"})
    original["_input_position"] = range(1, len(original) + 1)

    pred_df = pd.DataFrame(preds)
    if pred_df.empty:
        return original
    pred_df["_input_position"] = range(1, len(pred_df) + 1)

    # rank_batch returns sorted predictions without original identifiers, so we merge
    # by prediction order only after carrying the input position before sorting.
    raw_preds = prediction_service.predict_flights_batch(payloads)
    raw_pred_df = pd.DataFrame(raw_preds)
    raw_pred_df["_input_position"] = range(1, len(raw_pred_df) + 1)
    merged = original.merge(raw_pred_df, on="_input_position", how="left")
    merged = merged.sort_values("delay_probability", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    merged["delay_probability_pct"] = (merged["delay_probability"] * 100).round(1)
    return merged


def _batch_mode(t: dict[str, Any], model_available: bool) -> None:
    st.markdown(f'<div class="fr-section-title">{t["batch_title"]}</div>', unsafe_allow_html=True)
    st.write(t["batch_body"])
    st.caption(t["csv_cols"])

    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        uploaded = st.file_uploader(t["upload"], type=["csv"], disabled=not model_available)
    with c2:
        use_sample = st.button(t["sample"], disabled=not model_available)

    df: pd.DataFrame | None = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = _sample_batch()

    if df is not None and model_available:
        try:
            ranked = rank_dataframe(df)
            display_cols = [col for col in ["rank", "airline", "origin", "destination", "crs_dep_time", "delay_probability_pct", "risk_level"] if col in ranked.columns]
            st.dataframe(ranked[display_cols] if display_cols else ranked, width="stretch", hide_index=True)
            buffer = io.StringIO()
            ranked.to_csv(buffer, index=False)
            st.download_button(t["download"], data=buffer.getvalue(), file_name="flightrisk_ranked_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(str(exc))


def _model_details(t: dict[str, Any], lang: str) -> None:
    st.markdown(f'<div class="fr-section-title">{t["model_title"]}</div>', unsafe_allow_html=True)
    info = _safe_model_info()
    metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
    main_metrics = metrics.get("main_model", metrics) if isinstance(metrics, dict) else {}

    tab1, tab2, tab3, tab4 = st.tabs([t["main_model"], "Features", "Metrics", t["europe_title"]])
    with tab1:
        st.markdown(f"**{t['main_model_body']}**")
        st.write(t["target_body"])
        st.write("P(ArrDel15 = 1)")
    with tab2:
        st.write(t["features_body"])
        st.write("Leakage control: only schedule-time fields and train-fitted historical aggregates are available at inference.")
    with tab3:
        wanted = ["roc_auc", "pr_auc", "f1", "precision_at_top_10pct", "lift_at_top_10pct"]
        cols = st.columns(len(wanted))
        for col, key in zip(cols, wanted):
            value = main_metrics.get(key)
            label = key.replace("_", " ").title()
            col.metric(label, f"{value:.3f}" if isinstance(value, (float, int)) else "n/a")
    with tab4:
        st.markdown(f'<div class="fr-warning">{t["europe_body"]}</div>', unsafe_allow_html=True)
        try:
            summary = prediction_service.european_context_summary()
            st.json(summary, expanded=False)
        except Exception:
            st.caption("Europe context files are not available in this environment.")


def main() -> None:
    _inject_css()
    lang, t = _language_selector()
    model_available = prediction_service.is_model_available()
    _topbar(t, model_available)
    _hero(t, lang, model_available)
    _steps(t)
    _single_prediction(t, lang, model_available)
    _batch_mode(t, model_available)
    _model_details(t, lang)
    st.markdown(f'<div class="fr-footer">{t["footer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
