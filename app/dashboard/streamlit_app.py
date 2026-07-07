"""FlightRisk Streamlit UI.

English is the default language. The page is intentionally simple:
1) predict the probability of ArrDel15 for one scheduled flight,
2) optionally rank a batch of flights,
3) expose the ML details in clear expanders.
"""
from __future__ import annotations

import io
import sys
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
  --ink: #0d2944;
  --ink-2: #183d61;
  --blue: #268bd2;
  --blue-2: #62b6e8;
  --sky: #e7f5ff;
  --paper: #f7fbff;
  --panel: rgba(255,255,255,.82);
  --line: rgba(13,41,68,.12);
  --muted: #607892;
  --green: #2fa876;
  --amber: #d99a35;
  --red: #d85b65;
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  color: var(--ink);
  background:
    radial-gradient(circle at 18% 0%, rgba(98,182,232,.30), transparent 24%),
    radial-gradient(circle at 82% 12%, rgba(38,139,210,.15), transparent 30%),
    linear-gradient(180deg, #dcecf7 0%, #edf6fc 44%, #f8fbff 100%);
}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.main .block-container { max-width: 1180px; padding-top: .25rem; padding-bottom: 2.5rem; }
h1, h2, h3, h4, p, div, span, label { color: var(--ink); }
.stSelectbox label, .stFileUploader label, .stTextInput label, .stNumberInput label { color: var(--ink) !important; font-weight: 800; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: rgba(255,255,255,.92);
  border: 1px solid var(--line);
  color: var(--ink);
  border-radius: 10px;
}
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
  background: linear-gradient(180deg, #38a6e8, #257bc2) !important;
  color: #ffffff !important;
  border: 1px solid rgba(38,139,210,.25) !important;
  border-radius: 11px !important;
  font-weight: 850 !important;
  box-shadow: 0 8px 18px rgba(32,105,164,.16);
  min-height: 2.35rem;
}
[data-testid="stMetric"] {
  background: rgba(255,255,255,.86);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: .78rem;
  box-shadow: 0 8px 22px rgba(22,55,84,.07);
}
[data-testid="stMetric"] label, [data-testid="stMetric"] div { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,.82) !important;
  border: 1px dashed rgba(38,139,210,.35) !important;
  border-radius: 16px !important;
  min-height: 92px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(38,139,210,.10) !important;
  color: var(--ink) !important;
  border: 1px solid rgba(38,139,210,.20) !important;
  border-radius: 10px !important;
}
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; border:1px solid var(--line); }
[data-testid="stForm"] {
  background:rgba(255,255,255,.80);
  border:1px solid var(--line);
  border-radius:18px;
  padding:1rem;
  box-shadow:0 14px 32px rgba(20,65,101,.07);
}
.fr-topbar {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:.72rem .95rem; margin-top:.15rem;
  background:rgba(255,255,255,.76); border:1px solid var(--line);
  border-radius:16px; box-shadow:0 10px 24px rgba(19,55,85,.08); backdrop-filter: blur(12px);
  min-height:52px;
}
.fr-logo { display:flex; align-items:center; gap:.65rem; font-size:1.05rem; font-weight:950; letter-spacing:.13em; color:var(--ink); }
.fr-logo-mark { color:var(--blue); font-size:1.35rem; }
.fr-nav { display:flex; flex-wrap:wrap; gap:1rem; align-items:center; color:#355d7c; font-size:.86rem; font-weight:750; }
.fr-nav span { color:#355d7c; }
.fr-hero {
  position:relative; overflow:hidden; margin:.5rem 0 .85rem 0; padding:1.1rem 1.15rem;
  border-radius:22px; border:1px solid rgba(38,139,210,.18);
  background:
    linear-gradient(rgba(38,139,210,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(38,139,210,.045) 1px, transparent 1px),
    radial-gradient(circle at 82% 18%, rgba(98,182,232,.24), transparent 27%),
    linear-gradient(112deg, #f9fcff 0%, #eaf5fc 54%, #d4e8f7 100%);
  background-size: 48px 48px, 48px 48px, auto, auto;
  box-shadow:0 24px 54px rgba(19,55,85,.14);
}
.fr-pill { display:inline-flex; align-items:center; padding:.38rem .68rem; border-radius:999px; background:rgba(38,139,210,.10); border:1px solid rgba(38,139,210,.20); color:#176599; font-size:.72rem; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }
.fr-title { margin:.48rem 0 .38rem 0; font-size:clamp(1.75rem,2.8vw,3.05rem); line-height:1.04; font-weight:950; letter-spacing:-.052em; color:var(--ink); max-width:760px; }
.fr-title span { color:#1986c8; }
.fr-subtitle { max-width:720px; font-size:.96rem; line-height:1.55; color:#3f5e78; }
.fr-truth { margin-top:.75rem; padding:.76rem .9rem; border-radius:15px; border:1px solid var(--line); background:rgba(255,255,255,.64); }
.fr-muted { color:var(--muted); }
.fr-card {
  background:rgba(255,255,255,.84);
  border:1px solid var(--line); border-radius:20px; padding:.95rem; box-shadow:0 14px 34px rgba(20,65,101,.10);
}
.fr-card h3 { margin:.15rem 0 .6rem 0; font-size:1rem; letter-spacing:.02em; color:var(--ink); }
.fr-aircraft-card {
  position:relative; overflow:hidden; min-height:88px; border-radius:16px; margin-bottom:.68rem;
  background:linear-gradient(180deg, rgba(255,255,255,.76), rgba(234,246,255,.74));
  border:1px solid rgba(38,139,210,.18);
}
.fr-aircraft-card:before { content:""; position:absolute; left:-8%; right:-8%; bottom:28px; height:36px; background:linear-gradient(90deg, transparent, rgba(15,54,88,.12), transparent); transform:rotate(-2deg); }
.fr-aircraft { position:absolute; right:8%; top:14%; font-size:2.65rem; color:var(--blue); transform:rotate(-8deg); text-shadow:0 12px 28px rgba(22,89,146,.16); }
.fr-aircraft-copy { position:absolute; left:.85rem; bottom:.62rem; right:5.4rem; color:#355d7c; font-size:.76rem; line-height:1.34; }
.fr-aircraft-copy b { color:var(--ink); }
.fr-mini { color:var(--muted); font-size:.9rem; line-height:1.48; }
.fr-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.72rem; margin:.7rem 0 .95rem 0; }
.fr-step { background:rgba(255,255,255,.84); border:1px solid var(--line); border-radius:16px; padding:.85rem; box-shadow:0 8px 22px rgba(20,65,101,.055); }
.fr-step b { color:var(--ink); }
.fr-section-title { margin:1.05rem 0 .5rem 0; font-size:1.22rem; font-weight:950; letter-spacing:-.02em; color:var(--ink); }
.fr-form-heading {
  display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;
  padding:.9rem 1rem; border-radius:18px; background:rgba(255,255,255,.72);
  border:1px solid var(--line); box-shadow:0 12px 26px rgba(20,65,101,.06); margin:.35rem 0 .75rem 0;
}
.fr-form-heading h3 { margin:0; font-size:1.05rem; color:var(--ink); letter-spacing:-.01em; }
.fr-form-heading p { margin:.25rem 0 0 0; color:var(--muted); font-size:.9rem; line-height:1.45; }
.fr-form-badges { display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end; min-width:250px; }
.fr-badge { display:inline-flex; align-items:center; padding:.35rem .55rem; border-radius:999px; background:rgba(38,139,210,.09); border:1px solid rgba(38,139,210,.16); color:#176599; font-size:.72rem; font-weight:850; }
.fr-chart-note { margin-top:.5rem; padding:.7rem .85rem; border-radius:14px; background:rgba(38,139,210,.08); border:1px solid rgba(38,139,210,.14); color:#355d7c; }
.fr-uploader-card { padding:.95rem; border-radius:18px; background:rgba(255,255,255,.74); border:1px solid var(--line); box-shadow:0 12px 26px rgba(20,65,101,.06); }
.fr-tech-intro { padding:.9rem 1rem; border-radius:18px; background:rgba(255,255,255,.72); border:1px solid var(--line); color:#355d7c; }

[data-testid="stExpander"] {
  background: rgba(255,255,255,.76) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 22px rgba(20,65,101,.055) !important;
  overflow: hidden !important;
  margin-bottom: .55rem !important;
}
[data-testid="stExpander"] details summary {
  background: rgba(255,255,255,.90) !important;
  border-bottom: 1px solid rgba(13,41,68,.08) !important;
  min-height: 2.6rem !important;
}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary div {
  color: var(--ink) !important;
  font-weight: 850 !important;
}
.fr-prob-card {
  background: rgba(255,255,255,.82);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 12px 26px rgba(20,65,101,.06);
  margin-top: .55rem;
}
.fr-prob-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:.72rem; }
.fr-prob-head h3 { margin:0; font-size:1.08rem; color:var(--ink); }
.fr-prob-head span { color:var(--muted); font-size:.86rem; }
.fr-prob-row { display:grid; grid-template-columns: 190px 1fr 64px; align-items:center; gap:.75rem; margin:.65rem 0; }
.fr-prob-label { font-weight:850; color:var(--ink); font-size:.9rem; }
.fr-prob-track { height:15px; border-radius:999px; background:rgba(13,41,68,.08); overflow:hidden; border:1px solid rgba(13,41,68,.06); }
.fr-prob-fill { height:100%; border-radius:999px; background:linear-gradient(90deg, #62b6e8, #268bd2); }
.fr-prob-fill.delay { background:linear-gradient(90deg, #ffd38a, #d99a35); }
.fr-prob-value { font-weight:950; color:var(--ink); text-align:right; }
.fr-detail-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.65rem; margin:.7rem 0; }
.fr-detail-card { background:rgba(255,255,255,.76); border:1px solid var(--line); border-radius:14px; padding:.78rem; }
.fr-detail-card b { color:var(--ink); }
.fr-detail-card span { color:var(--muted); font-size:.9rem; line-height:1.45; }
.fr-flow { display:flex; align-items:stretch; gap:.45rem; flex-wrap:wrap; margin:.75rem 0; }
.fr-flow-step { flex:1 1 150px; min-width:145px; background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:14px; padding:.72rem; }
.fr-flow-step b { display:block; color:var(--ink); font-size:.9rem; margin-bottom:.15rem; }
.fr-flow-step span { color:var(--muted); font-size:.82rem; line-height:1.35; }
.fr-flow-arrow { display:flex; align-items:center; color:#268bd2; font-weight:950; }
.fr-command {
  background: rgba(13,41,68,.055);
  border: 1px solid rgba(13,41,68,.10);
  border-radius: 12px;
  padding: .65rem .75rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--ink);
  font-size: .86rem;
  overflow-x: auto;
}
@media (max-width: 900px) { .fr-prob-row { grid-template-columns:1fr; gap:.25rem; } .fr-prob-value { text-align:left; } .fr-detail-grid { grid-template-columns:1fr; } .fr-flow-arrow { display:none; } }

.fr-footer { margin-top:1.2rem; color:var(--muted); font-size:.86rem; text-align:center; }
@media (max-width: 900px) { .fr-strip { grid-template-columns:1fr; } .fr-form-heading { flex-direction:column; } .fr-form-badges { justify-content:flex-start; min-width:0; } }
</style>
"""

TEXT = {
    "en": {
        "nav": ["Predict", "Batch", "Model details"],
        "status_loaded": "Model loaded",
        "status_missing": "Model missing",
        "hero_pill": "Flight delay probability model",
        "hero_title": "Will this flight arrive <span>15+ minutes late?</span>",
        "hero_sub": "FlightRisk estimates, before departure, the probability that a scheduled flight arrives 15+ minutes late. Batch mode can then sort many flights by predicted delay probability.",
        "truth_title": "What the model predicts",
        "truth_body": "Target: P(ArrDel15 = 1). ArrDel15 means the arrival delay is 15 minutes or more.",
        "visual_title": "Pre-departure delay risk",
        "visual_body": "A schedule-time model for estimating arrival delay probability before departure.",
        "example_title": "Example prediction",
        "prob": "Delay probability",
        "bucket": "Risk level",
        "target": "Target",
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
        "drivers": "Main signals",
        "chart_title": "Prediction profile",
        "chart_note": "This chart shows the model output as a probability split, not a guarantee.",
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
        "tech_title": "Technical model details",
        "tech_intro": "Expandable portfolio documentation: model card, variables, cleaning, model candidates, architecture and pipeline.",
        "footer": "Portfolio ML project. Not for safety-critical dispatch, legal, compensation or operational aviation decisions.",
    },
    "es": {
        "nav": ["Predecir", "Batch", "Detalles del modelo"],
        "status_loaded": "Modelo cargado",
        "status_missing": "Modelo no disponible",
        "hero_pill": "Modelo de probabilidad de retraso",
        "hero_title": "¿Llegará este vuelo con <span>15+ minutos de retraso?</span>",
        "hero_sub": "FlightRisk estima, antes de la salida, la probabilidad de que un vuelo programado llegue con 15 minutos o más de retraso. El modo batch ordena muchos vuelos por probabilidad estimada.",
        "truth_title": "Qué predice el modelo",
        "truth_body": "Target: P(ArrDel15 = 1). ArrDel15 significa que el retraso de llegada es de 15 minutos o más.",
        "visual_title": "Riesgo antes de salida",
        "visual_body": "Modelo basado en horarios programados para estimar retraso antes de la salida.",
        "example_title": "Ejemplo de predicción",
        "prob": "Probabilidad de retraso",
        "bucket": "Nivel de riesgo",
        "target": "Target",
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
        "drivers": "Señales principales",
        "chart_title": "Perfil de predicción",
        "chart_note": "Este gráfico muestra la salida del modelo como reparto de probabilidad, no como garantía.",
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
        "tech_title": "Detalles técnicos del modelo",
        "tech_intro": "Documentación portfolio expandible: model card, variables, limpieza, modelos probados, arquitectura y pipeline.",
        "footer": "Proyecto portfolio ML. No usar para dispatch crítico, decisiones legales, compensación o aviación operacional.",
    },
}


def _inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _language_selector(column) -> tuple[str, dict[str, Any]]:
    with column:
        choice = st.selectbox("Language / Idioma", ["English", "Español"], index=0, key="language_selector", label_visibility="collapsed")
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


def _topbar(t: dict[str, Any], model_available: bool) -> None:
    status = t["status_loaded"] if model_available else t["status_missing"]
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


def _risk_label(level: str, lang: str) -> str:
    mapping_en = {"low": "Low", "moderate": "Moderate", "high": "High"}
    mapping_es = {"low": "Bajo", "moderate": "Moderado", "high": "Alto"}
    return (mapping_en if lang == "en" else mapping_es).get(level, level)


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


def _sample_prediction(model_available: bool) -> dict[str, Any]:
    fallback = {"delay_probability": 0.238, "risk_level": "moderate", "top_factors": ["sample route profile"]}
    if not model_available:
        return fallback
    try:
        return prediction_service.predict_flight(_sample_input())
    except Exception:
        return fallback


def _aircraft_visual(t: dict[str, Any]) -> None:
    st.markdown(
        f"""
<div class="fr-aircraft-card" aria-hidden="true">
  <div class="fr-aircraft">✈</div>
  <div class="fr-aircraft-copy"><b>{t['visual_title']}</b><br>{t['visual_body']}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _hero(t: dict[str, Any], lang: str, model_available: bool) -> None:
    result = _sample_prediction(model_available)
    prob = float(result.get("delay_probability", 0.238)) * 100
    risk = _risk_label(str(result.get("risk_level", "moderate")), lang)

    st.markdown('<section class="fr-hero">', unsafe_allow_html=True)
    left, right = st.columns([0.54, 0.46], gap="large")
    with left:
        st.markdown(
            f"""
<div class="fr-pill">{t['hero_pill']}</div>
<div class="fr-title">{t['hero_title']}</div>
<div class="fr-subtitle">{t['hero_sub']}</div>
<div class="fr-truth"><b>{t['truth_title']}</b><br><span class="fr-muted">{t['truth_body']}</span></div>
""",
            unsafe_allow_html=True,
        )
    with right:
        _aircraft_visual(t)
        st.markdown(f"**{t['example_title']}**")
        st.caption("DL · JFK → LAX · 18:30 · 375 min · 2,475 mi")
        c1, c2, c3 = st.columns(3)
        c1.metric(t["prob"], f"{prob:.1f}%")
        c2.metric(t["bucket"], risk)
        c3.metric(t["target"], "ArrDel15")
        st.caption("P(ArrDel15 = 1) · probability of arriving 15+ minutes late")
    st.markdown('</section>', unsafe_allow_html=True)


def _steps(t: dict[str, Any]) -> None:
    html = '<div class="fr-strip">'
    for title, body in t["steps"]:
        html += f'<div class="fr-step"><b>{title}</b><br><span class="fr-muted">{body}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _prediction_payload_from_form(t: dict[str, Any], prefix: str = "single") -> PredictionInput:
    c1, c2, c3 = st.columns(3)
    with c1:
        airline = st.text_input(t["airline"], "DL", key=f"{prefix}_airline").upper()
        month = st.number_input(t["month"], min_value=1, max_value=12, value=7, step=1, key=f"{prefix}_month")
        dep = st.number_input(t["dep"], min_value=0, max_value=2400, value=1830, step=5, key=f"{prefix}_dep")
    with c2:
        origin = st.text_input(t["origin"], "JFK", key=f"{prefix}_origin").upper()
        dow = st.number_input(t["dow"], min_value=1, max_value=7, value=5, step=1, key=f"{prefix}_dow")
        arr = st.number_input(t["arr"], min_value=0, max_value=2400, value=2145, step=5, key=f"{prefix}_arr")
    with c3:
        dest = st.text_input(t["dest"], "LAX", key=f"{prefix}_dest").upper()
        elapsed = st.number_input(t["elapsed"], min_value=1, value=375, step=5, key=f"{prefix}_elapsed")
        distance = st.number_input(t["distance"], min_value=1.0, value=2475.0, step=10.0, key=f"{prefix}_distance")
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


def _probability_chart_html(probability: float, lang: str) -> str:
    pct_delay = max(0.0, min(100.0, round(probability * 100, 1)))
    pct_no_delay = round(100 - pct_delay, 1)
    on_time_label = "Arrives <15 min late" if lang == "en" else "Llega con <15 min de retraso"
    delay_label = "Arrives 15+ min late" if lang == "en" else "Llega con 15+ min de retraso"
    subtitle = "Model output split" if lang == "en" else "Reparto de salida del modelo"
    return f"""
<div class="fr-prob-card">
  <div class="fr-prob-head"><h3>{'Prediction profile' if lang == 'en' else 'Perfil de predicción'}</h3><span>{subtitle}</span></div>
  <div class="fr-prob-row">
    <div class="fr-prob-label">{delay_label}</div>
    <div class="fr-prob-track"><div class="fr-prob-fill delay" style="width:{pct_delay}%"></div></div>
    <div class="fr-prob-value">{pct_delay:.1f}%</div>
  </div>
  <div class="fr-prob-row">
    <div class="fr-prob-label">{on_time_label}</div>
    <div class="fr-prob-track"><div class="fr-prob-fill" style="width:{pct_no_delay}%"></div></div>
    <div class="fr-prob-value">{pct_no_delay:.1f}%</div>
  </div>
</div>
"""


def _single_prediction(t: dict[str, Any], lang: str, model_available: bool) -> None:
    st.markdown(
        f"""
<div class="fr-form-heading">
  <div><h3>{t['input_title']}</h3><p>{t['input_help']}</p></div>
  <div class="fr-form-badges"><span class="fr-badge">Pre-flight only</span><span class="fr-badge">ArrDel15 target</span><span class="fr-badge">BTS trained</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.form("single_prediction_form"):
        payload = _prediction_payload_from_form(t, "single")
        submitted = st.form_submit_button(t["predict"], disabled=not model_available)

    if submitted and model_available:
        result = prediction_service.predict_flight(payload)
        prob = float(result.get("delay_probability", 0.0))
        risk = _risk_label(str(result.get("risk_level", "")), lang)
        c1, c2, c3 = st.columns(3)
        c1.metric(t["prob"], f"{prob * 100:.1f}%")
        c2.metric(t["bucket"], risk)
        c3.metric(t["target"], "ArrDel15")

        st.markdown(_probability_chart_html(prob, lang), unsafe_allow_html=True)
        st.markdown(f'<div class="fr-chart-note">{t["chart_note"]}</div>', unsafe_allow_html=True)

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
    payloads = _payloads_from_df(df)
    original = df.reset_index(drop=False).rename(columns={"index": "_original_index"})
    original["_input_position"] = range(1, len(original) + 1)
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
    st.markdown(f'<div class="fr-uploader-card"><b>{t["batch_body"]}</b><br><span class="fr-muted">{t["csv_cols"]}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.70, 0.30], gap="large")
    with c1:
        uploaded = st.file_uploader(t["upload"], type=["csv"], disabled=not model_available)
    with c2:
        st.write("")
        st.write("")
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
            if "risk_level" in ranked.columns:
                risk_counts = ranked["risk_level"].value_counts().rename_axis("Risk level").reset_index(name="Flights")
                st.bar_chart(risk_counts.set_index("Risk level"), height=240)
            buffer = io.StringIO()
            ranked.to_csv(buffer, index=False)
            st.download_button(t["download"], data=buffer.getvalue(), file_name="flightrisk_ranked_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(str(exc))


def _metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    return f"{value:.3f}" if isinstance(value, (float, int)) else "n/a"


def _technical_details(t: dict[str, Any]) -> None:
    st.markdown(f'<div class="fr-section-title">{t["tech_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fr-tech-intro">{t["tech_intro"]}</div>', unsafe_allow_html=True)
    info = _safe_model_info()
    card = _safe_model_card()
    metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
    main_metrics = metrics.get("main_model", metrics) if isinstance(metrics, dict) else {}
    selected_model = card.get("selected_model") or info.get("model_name") or "selected validation candidate"
    threshold = info.get("decision_threshold", card.get("decision_threshold", "n/a"))

    with st.expander("1. Model card", expanded=True):
        st.markdown(
            f"""
**Task:** binary classification.  
**Prediction:** `P(ArrDel15 = 1)`.  
**Meaning:** probability that a scheduled flight arrives **15+ minutes late**.  
**Selected model:** `{selected_model}`.  
**Decision threshold:** `{threshold}`.  
**Intended use:** portfolio ML demo for schedule-time delay probability estimation.  
**Not intended for:** safety-critical dispatch, legal, compensation or operational aviation decisions.
"""
        )

    with st.expander("2. Variables used"):
        st.markdown(
            """
**Raw schedule inputs**
- `Reporting_Airline`
- `Origin`
- `Dest`
- `Month`
- `DayofMonth` / `FlightDate` when available
- `DayOfWeek`
- `CRSDepTime`
- `CRSArrTime`
- `CRSElapsedTime`
- `Distance`

**Engineered pre-flight features**
- departure hour and arrival hour
- route key
- weekend flag
- time-of-day periods
- red-eye / peak-time flags
- distance band
- scheduled speed

**Historical aggregate features**
- carrier historical delay rate
- route historical delay rate
- origin and destination delay rates
- carrier-route delay rate
- origin-hour / destination-hour rates
- carrier-departure-hour rate
"""
        )

    with st.expander("3. Cleaning and leakage controls"):
        st.markdown(
            """
**Cleaning**
- Normalize BTS column names across monthly CSVs.
- Drop rows with missing `ArrDel15` target.
- Filter cancelled/diverted rows when present.
- Remove forbidden leakage columns before training.

**Leakage controls**
- No actual arrival/departure time is used.
- No `ArrDelay`, `DepDelay`, `TaxiOut`, `TaxiIn`, `WheelsOff`, `WheelsOn`, `AirTime` or delay-cause columns are used as features.
- `Cancelled` and `Diverted` are cleaning filters, not model inputs.
- Historical aggregate rates are fitted from training data and use fallbacks for unseen entities.
"""
        )

    with st.expander("4. Models tested and selection"):
        st.markdown(
            """
**Candidate models**
- Logistic Regression baseline
- L1 Logistic Regression
- RandomForestClassifier
- ExtraTreesClassifier
- Optional GradientBoostingClassifier

**Selection logic**
- Models are compared on validation metrics.
- For the probability-first version, PR-AUC is the recommended selection metric.
- Ranking metrics such as Precision@Top10% and Lift@Top10% are reported as secondary product metrics.
"""
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", _metric_value(main_metrics, "roc_auc"))
        c2.metric("PR-AUC", _metric_value(main_metrics, "pr_auc"))
        c3.metric("F1", _metric_value(main_metrics, "f1"))
        c4.metric("Lift@10%", _metric_value(main_metrics, "lift_at_top_10pct"))

    with st.expander("5. Architecture"):
        st.markdown(
            """
<div class="fr-detail-grid">
  <div class="fr-detail-card"><b>Streamlit UI</b><br><span>Single-flight prediction, batch scoring and portfolio documentation.</span></div>
  <div class="fr-detail-card"><b>Prediction service</b><br><span>Loads the versioned model artifact and exposes prediction helpers.</span></div>
  <div class="fr-detail-card"><b>Model artifact</b><br><span>Serialized scikit-learn preprocessing and classifier pipeline.</span></div>
  <div class="fr-detail-card"><b>Reports</b><br><span>Metrics, feature importance, confusion matrix and error analysis.</span></div>
  <div class="fr-detail-card"><b>Training scripts</b><br><span>Reproducible commands for sampling, training and temporal backtesting.</span></div>
  <div class="fr-detail-card"><b>FastAPI service</b><br><span>API layer for deployment and integration beyond the Streamlit UI.</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Main folders**")
        st.markdown(
            """
- `app/dashboard/` — Streamlit interface
- `app/api/` — FastAPI service
- `app/services/` — prediction service layer
- `src/data/` — loading, cleaning and splitting
- `src/features/` — engineered and historical features
- `src/models/` — training, evaluation, thresholding and registry
- `scripts/` — reproducible training and backtesting commands
"""
        )

    with st.expander("6. Training pipeline"):
        st.markdown(
            """
<div class="fr-flow">
  <div class="fr-flow-step"><b>BTS monthly CSVs</b><span>Raw schedule and target columns.</span></div><div class="fr-flow-arrow">→</div>
  <div class="fr-flow-step"><b>Clean data</b><span>Normalize columns, remove missing target and leakage fields.</span></div><div class="fr-flow-arrow">→</div>
  <div class="fr-flow-step"><b>Build features</b><span>Schedule-time features plus train-fitted historical rates.</span></div><div class="fr-flow-arrow">→</div>
  <div class="fr-flow-step"><b>Train candidates</b><span>Compare baseline and tree-based models.</span></div><div class="fr-flow-arrow">→</div>
  <div class="fr-flow-step"><b>Evaluate & save</b><span>Store artifact, threshold, metrics and reports.</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Training command example**")
        st.markdown(
            '<div class="fr-command">python -m scripts.run_real_data_demo --selection-metric pr_auc --bootstrap-samples 0 --max-rows-per-month 50000</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    _inject_css()
    left, right = st.columns([0.78, 0.22], gap="small")
    lang, t = _language_selector(right)
    model_available = prediction_service.is_model_available()
    with left:
        _topbar(t, model_available)
    _hero(t, lang, model_available)
    _steps(t)
    _single_prediction(t, lang, model_available)
    _batch_mode(t, model_available)
    _technical_details(t)
    st.markdown(f'<div class="fr-footer">{t["footer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
