"""Bilingual (English / Spanish) copy for the FlightRisk Streamlit cockpit.

This module intentionally holds *only* text and a couple of tiny lookup
helpers -- no Streamlit imports and no rendering logic -- so the copy can be
read, extended or sanity-checked on its own, independently of
``streamlit_app.py``.

Usage from the dashboard::

    from app.dashboard.i18n import translator, translate_backend_string

    LANG = "en"  # or "es", chosen via the language toggle
    t = translator(LANG)
    st.markdown(t("hero_title"))
    st.markdown(t("note_us", origin="JFK", destination="LAX", threshold=0.43))
"""
from __future__ import annotations

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "es")

# ---------------------------------------------------------------------------
# UI copy authored for this dashboard.
#
# Each entry maps a stable copy key -> {"en": ..., "es": ...}. Templates may
# contain `{placeholder}` / `{placeholder:.2f}` style fields; callers pass
# matching keyword arguments to `t(key, **kwargs)` and Python's own
# `str.format` resolves the format spec, exactly like an f-string would.
# ---------------------------------------------------------------------------
TRANSLATIONS: dict[str, dict[str, str]] = {
    # --- Hero ------------------------------------------------------------
    "hero_kicker": {
        "en": "FlightRisk v6.6 · real aviation data",
        "es": "FlightRisk v6.6 · datos reales de aviación",
    },
    "hero_title": {
        "en": "Flight delay risk ranking.",
        "es": "Ranking de riesgo de retraso de vuelos.",
    },
    "hero_sub": {
        "en": "A real-data ML system for ranking delay risk across U.S. flights and European punctuality patterns.",
        "es": "Un sistema de ML con datos reales para priorizar el riesgo de retraso en vuelos de EE. UU. y en patrones de puntualidad europeos.",
    },
    "hero_chip_bts": {"en": "BTS flight records", "es": "Registros de vuelo BTS"},
    "hero_chip_caa": {"en": "UK CAA punctuality", "es": "Puntualidad UK CAA"},
    "hero_chip_fastapi": {"en": "FastAPI", "es": "FastAPI"},
    "hero_chip_monitoring": {"en": "Monitoring", "es": "Monitorización"},

    # --- Intro two-column (What it does / System summary) ---------------
    "overview_kicker": {"en": "FlightRisk v6.6", "es": "FlightRisk v6.6"},
    "overview_title": {"en": "What it does.", "es": "Qué hace."},
    "overview_sub": {
        "en": "Ranks scheduled flight delay risk using real aviation datasets, then serves the result through an API and a lightweight cockpit UI.",
        "es": "Prioriza el riesgo de retraso de vuelos programados usando datos reales de aviación, y sirve el resultado a través de una API y una cabina de predicción ligera.",
    },
    "overview_chip_us": {"en": "U.S. BTS data", "es": "Datos BTS (EE. UU.)"},
    "overview_chip_eu": {"en": "UK CAA data", "es": "Datos UK CAA"},
    "overview_chip_lift": {"en": "Lift@10", "es": "Lift@10"},
    "overview_chip_telemetry": {"en": "Telemetry", "es": "Telemetría"},

    "summary_micro": {"en": "System summary", "es": "Resumen del sistema"},
    "summary_title": {
        "en": "Built as an ML engineering demo, not a BI dashboard",
        "es": "Construido como una demo de ingeniería de ML, no como un dashboard de BI",
    },
    "summary_b1": {
        "en": "<b>U.S. path:</b> ranks individual scheduled flights using real BTS flight-level records.",
        "es": "<b>Ruta EE. UU.:</b> prioriza vuelos programados individuales usando registros reales de BTS a nivel de vuelo.",
    },
    "summary_b2": {
        "en": "<b>Europe path:</b> models aggregate punctuality patterns from real UK CAA data.",
        "es": "<b>Ruta Europa:</b> modela patrones agregados de puntualidad a partir de datos reales de UK CAA.",
    },
    "summary_b3": {
        "en": "<b>ML workflow:</b> leakage-safe features, time-aware validation, model selection and Lift@10 reporting.",
        "es": "<b>Flujo de ML:</b> variables sin fuga de datos, validación consciente del tiempo, selección de modelo y reporte de Lift@10.",
    },
    "summary_b4": {
        "en": "<b>Serving:</b> FastAPI endpoints, Streamlit cockpit, prediction logging, drift checks and model metadata.",
        "es": "<b>Despliegue:</b> endpoints de FastAPI, cabina en Streamlit, registro de predicciones, chequeos de drift y metadatos del modelo.",
    },

    "warn_no_model": {
        "en": "No trained U.S. model artifact found yet. Run `python -m scripts.run_real_data_demo --max-rows 200000` with BTS data, then reload this page.",
        "es": "Todavía no hay un modelo entrenado para EE. UU. Ejecuta `python -m scripts.run_real_data_demo --max-rows 200000` con datos BTS y vuelve a cargar esta página.",
    },

    # --- Tabs --------------------------------------------------------------
    "tab_predict": {"en": "✈️ Prediction cockpit", "es": "✈️ Cabina de predicción"},
    "tab_about": {"en": "📖 About the project", "es": "📖 Sobre el proyecto"},
    "tab_model": {"en": "🧠 Model intelligence", "es": "🧠 Inteligencia del modelo"},
    "tab_system": {"en": "📡 Live system", "es": "📡 Sistema en vivo"},

    # --- Predict tab: stat bar / domain toggle -----------------------------
    "statbar_us_label": {"en": "U.S. data", "es": "Datos EE. UU."},
    "statbar_us_value": {"en": "BTS flight-level · real", "es": "BTS a nivel de vuelo · real"},
    "statbar_eu_label": {"en": "Europe data", "es": "Datos Europa"},
    "statbar_eu_value": {"en": "UK CAA aggregate · real", "es": "UK CAA agregado · real"},
    "statbar_metric_label": {"en": "Ranking metric", "es": "Métrica de ranking"},
    "statbar_metric_value": {"en": "Lift@10 + PR-AUC", "es": "Lift@10 + PR-AUC"},

    "domain_label": {"en": "Flight domain", "es": "Ámbito del vuelo"},
    "domain_us": {"en": "US flight", "es": "Vuelo en EE. UU."},
    "domain_eu": {"en": "Europe experimental", "es": "Europa (experimental)"},

    "data_status_prefix": {
        "en": "<b>Data status:</b> Europe context = {status} · rows = {rows} · routes = {routes}",
        "es": "<b>Estado de los datos:</b> contexto de Europa = {status} · filas = {rows} · rutas = {routes}",
    },
    "eu_status_real": {"en": "REAL CAA", "es": "CAA REAL"},
    "eu_status_sample": {"en": "sample", "es": "muestra"},
    "eu_status_none": {"en": "not available", "es": "no disponible"},

    # --- Predict tab: route setup card --------------------------------------
    "route_setup_title": {"en": "Route setup", "es": "Configuración de la ruta"},
    "route_setup_copy": {
        "en": "Choose a scenario or override the schedule-time fields. Preset state is locked to the selected route, so the UI doesn't drift into mismatched airport combinations.",
        "es": "Elige un escenario o modifica los campos de horario. El estado de cada preset queda fijado a la ruta elegida, para que la UI no termine mezclando combinaciones de aeropuertos que no encajan.",
    },
    "scenario_preset_us_label": {"en": "Scenario preset", "es": "Escenario preestablecido"},
    "scenario_preset_eu_label": {"en": "European scenario", "es": "Escenario europeo"},

    "preset_us_transcon": {
        "en": "JFK → LAX · transcontinental evening",
        "es": "JFK → LAX · vuelo transcontinental nocturno",
    },
    "preset_us_business": {
        "en": "ATL → ORD · business corridor",
        "es": "ATL → ORD · corredor de negocios",
    },
    "preset_us_westcoast": {
        "en": "SFO → SEA · west coast hop",
        "es": "SFO → SEA · salto en la costa oeste",
    },
    "preset_eu_bcnams": {
        "en": "Barcelona → Amsterdam · European tech corridor",
        "es": "Barcelona → Amsterdam · corredor tecnológico europeo",
    },
    "preset_eu_madlhr": {
        "en": "Madrid → London Heathrow · capital link",
        "es": "Madrid → London Heathrow · conexión entre capitales",
    },
    "preset_eu_frafco": {
        "en": "Frankfurt → Rome · hub connection",
        "es": "Frankfurt → Rome · conexión de hub",
    },

    "field_carrier": {"en": "Carrier", "es": "Aerolínea"},
    "field_origin": {"en": "Origin", "es": "Origen"},
    "field_destination": {"en": "Destination", "es": "Destino"},
    "field_distance_miles": {"en": "Distance (miles)", "es": "Distancia (millas)"},
    "field_month": {"en": "Month", "es": "Mes"},
    "field_day": {"en": "Day", "es": "Día"},
    "field_dep_hhmm": {"en": "Dep HHMM", "es": "Salida HHMM"},
    "field_arr_hhmm": {"en": "Arr HHMM", "es": "Llegada HHMM"},
    "field_duration_min": {"en": "Scheduled duration (minutes)", "es": "Duración programada (minutos)"},

    "field_airline_eu": {"en": "Airline", "es": "Aerolínea"},
    "field_auto_distance": {"en": "Auto-estimate route distance", "es": "Estimar distancia automáticamente"},
    "field_manual_distance": {"en": "Manual distance (miles)", "es": "Distancia manual (millas)"},

    "run_button": {"en": "Run prediction", "es": "Ejecutar predicción"},
    "error_hhmm": {
        "en": "Scheduled times must be valid HHMM values, e.g. 1830 or 2400 for midnight.",
        "es": "Los horarios programados deben ser valores HHMM válidos, por ejemplo 1830 o 2400 para medianoche.",
    },

    "eu_no_match_caption": {
        "en": "No aggregate CAA match for this exact route/month. Route inference still works, but UK CAA aggregate context is missing for this profile.",
        "es": "No hay coincidencia agregada de la CAA para esta ruta/mes exactos. La inferencia de ruta sigue funcionando, pero falta el contexto agregado de UK CAA para este perfil.",
    },
    "metric_caa_match": {"en": "CAA match", "es": "Coincidencia CAA"},
    "metric_pct_late": {"en": "15+ min late", "es": "15+ min de retraso"},
    "metric_avg_delay": {"en": "Avg delay", "es": "Retraso medio"},

    # --- Route map visual ----------------------------------------------------
    "route_origin_label": {"en": "Origin", "es": "Origen"},
    "route_destination_label": {"en": "Destination", "es": "Destino"},
    "route_inference_suffix": {"en": "route inference", "es": "inferencia de ruta"},
    "route_schedule_only": {"en": "Schedule-only input", "es": "Solo datos de horario"},
    "route_leakage_safe": {"en": "Leakage-safe", "es": "Sin fuga de datos"},
    "route_distance_prefix": {"en": "Distance", "es": "Distancia"},
    "distance_auto_pending": {"en": "auto distance pending", "es": "distancia automática pendiente"},

    # --- Result visual ---------------------------------------------------
    "result_delay_risk_label": {"en": "Delay risk", "es": "Riesgo de retraso"},
    "result_prediction_complete": {"en": "Prediction complete", "es": "Predicción completada"},
    "result_risk_suffix": {"en": "RISK", "es": "RIESGO"},
    "risk_low": {"en": "LOW", "es": "BAJO"},
    "risk_moderate": {"en": "MODERATE", "es": "MODERADO"},
    "risk_high": {"en": "HIGH", "es": "ALTO"},

    "note_us": {
        "en": "U.S. schedule-time inference for {origin} → {destination}. Tuned threshold: {threshold:.2f}.",
        "es": "Inferencia por horario programado para {origin} → {destination} (EE. UU.). Umbral ajustado: {threshold:.2f}.",
    },
    "note_eu_matched": {
        "en": "European punctuality path for {origin_label} → {destination_label}. CAA context: {pct:.1%} 15+ min late · {avg:.1f} min avg delay.",
        "es": "Ruta de puntualidad europea para {origin_label} → {destination_label}. Contexto CAA: {pct:.1%} con 15+ min de retraso · {avg:.1f} min de retraso medio.",
    },
    "note_eu_nomatch": {
        "en": "European punctuality path for {origin_label} → {destination_label}. No UK CAA aggregate match is available for this exact route/month.",
        "es": "Ruta de puntualidad europea para {origin_label} → {destination_label}. No hay una coincidencia agregada de UK CAA para esta ruta/mes exactos.",
    },

    "signal_drivers_title": {"en": "Signal drivers", "es": "Factores de riesgo"},
    "signal_drivers_caption": {
        "en": "These are descriptive associations from the model's inputs, not proven causes of delay.",
        "es": "Son asociaciones descriptivas a partir de las variables del modelo, no causas probadas del retraso.",
    },
    "structured_output_title": {"en": "Structured output", "es": "Salida estructurada"},
    "structured_output_caption": {
        "en": "Raw response shape returned by the prediction service (same fields the API returns).",
        "es": "Forma cruda de la respuesta del servicio de predicción (los mismos campos que devuelve la API).",
    },

    # --- Model intelligence tab -------------------------------------------
    "model_capsule_title": {"en": "Model intelligence capsule", "es": "Cápsula de inteligencia del modelo"},
    "metric_rocauc": {"en": "ROC-AUC", "es": "ROC-AUC"},
    "metric_prauc": {"en": "PR-AUC", "es": "PR-AUC"},
    "metric_f1": {"en": "F1", "es": "F1"},
    "metric_threshold": {"en": "Threshold", "es": "Umbral"},
    "help_rocauc": {
        "en": "How well the model ranks a random late flight above a random on-time flight, across all thresholds (0.5 = random, 1.0 = perfect).",
        "es": "Qué tan bien el modelo ordena un vuelo con retraso al azar por encima de uno puntual al azar, en todos los umbrales (0.5 = azar, 1.0 = perfecto).",
    },
    "help_prauc": {
        "en": "Precision vs. recall trade-off; more informative than ROC-AUC when delayed flights are the minority class.",
        "es": "Balance entre precisión y recall; más informativo que ROC-AUC cuando los vuelos con retraso son la clase minoritaria.",
    },
    "help_f1": {
        "en": "Harmonic mean of precision and recall at the tuned decision threshold.",
        "es": "Media armónica entre precisión y recall en el umbral de decisión ajustado.",
    },
    "help_threshold": {
        "en": "Probability cutoff above which a flight is classified as high risk; tuned on validation data for the best F1.",
        "es": "Umbral de probabilidad por encima del cual un vuelo se clasifica como de alto riesgo; ajustado con datos de validación para maximizar F1.",
    },
    "caption_selected_model": {
        "en": "Selected model: {model} · train rows: {train} · validation rows: {val} · test rows: {test}",
        "es": "Modelo seleccionado: {model} · filas de entrenamiento: {train} · filas de validación: {val} · filas de test: {test}",
    },
    "eu_context_layer_title": {"en": "European context layer", "es": "Capa de contexto europeo"},
    "metric_context_rows": {"en": "Context rows", "es": "Filas de contexto"},
    "metric_eu_routes": {"en": "European routes", "es": "Rutas europeas"},
    "metric_source_mode": {"en": "Source mode", "es": "Modo de fuente"},
    "value_real_caa": {"en": "REAL CAA", "es": "CAA REAL"},
    "value_missing": {"en": "missing", "es": "no disponible"},
    "eu_context_caption": {
        "en": "This is an aggregated route/month punctuality layer. If `europe_punctuality_context.csv` exists it uses your generated UK CAA context; otherwise European prediction is blocked until real data is generated.",
        "es": "Esta es una capa agregada de puntualidad por ruta/mes. Si existe `europe_punctuality_context.csv` se usa tu contexto generado de UK CAA; si no, la predicción europea queda bloqueada hasta que generes datos reales.",
    },
    "expander_full_metadata": {"en": "Full model metadata", "es": "Metadatos completos del modelo"},

    # --- Live system tab ---------------------------------------------------
    "system_title": {"en": "Live system telemetry", "es": "Telemetría del sistema en vivo"},
    "metric_logged_predictions": {"en": "Logged predictions", "es": "Predicciones registradas"},
    "metric_avg_probability": {"en": "Average probability", "es": "Probabilidad media"},
    "metric_high_risk_share": {"en": "High-risk share", "es": "Proporción de alto riesgo"},
    "metric_drift_status": {"en": "Drift status", "es": "Estado del drift"},
    "help_logged_predictions": {
        "en": "Total predictions written to the local monitoring log since it was created.",
        "es": "Total de predicciones escritas en el registro de monitorización local desde su creación.",
    },
    "help_avg_probability": {
        "en": "Mean predicted delay probability across all logged predictions.",
        "es": "Probabilidad media de retraso predicha en todas las predicciones registradas.",
    },
    "help_high_risk_share": {
        "en": "Share of logged predictions classified as high risk.",
        "es": "Proporción de predicciones registradas clasificadas como de alto riesgo.",
    },
    "help_drift_status": {
        "en": "Population Stability Index comparison between recent predictions and the training reference distribution.",
        "es": "Comparación mediante el Índice de Estabilidad Poblacional (PSI) entre las predicciones recientes y la distribución de referencia de entrenamiento.",
    },
    "system_caption": {
        "en": "Monitoring is logged locally for demo purposes. Drift is a lightweight PSI check against the training reference distribution.",
        "es": "La monitorización se registra localmente con fines de demostración. El drift es un chequeo PSI ligero frente a la distribución de referencia de entrenamiento.",
    },
    "expander_payloads": {"en": "Monitoring payloads", "es": "Payloads de monitorización"},

    "footer_caption": {
        "en": "Data sources: U.S. DOT/BTS flight-level records and UK CAA aggregate punctuality data.",
        "es": "Fuentes de datos: registros a nivel de vuelo de U.S. DOT/BTS y datos agregados de puntualidad de UK CAA.",
    },

    # --- About tab (portfolio case study) ---------------------------------
    "about_kicker": {"en": "Portfolio case study", "es": "Caso de portfolio"},
    "about_title": {"en": "How FlightRisk was built.", "es": "Cómo se construyó FlightRisk."},
    "about_lede": {
        "en": "FlightRisk estimates, before a flight departs, how likely it is to arrive 15 or more minutes late — using only information available at scheduling time. It's built as an educational ML engineering portfolio project: real data, honest scope, and decisions documented the way an interviewer would probe them.",
        "es": "FlightRisk estima, antes de que un vuelo despegue, la probabilidad de que llegue con 15 minutos o más de retraso, usando solo la información disponible en el momento de programar el vuelo. Está construido como un proyecto educativo de portfolio en ingeniería de ML: datos reales, alcance honesto y decisiones documentadas tal como las cuestionaría un entrevistador.",
    },

    "whats_real_title": {"en": "What's real, and what's a demo", "es": "Qué es real y qué es una demo"},
    "whats_real_us_title": {"en": "U.S. path — production-shaped", "es": "Ruta EE. UU. — con forma de producción"},
    "whats_real_us_copy": {
        "en": "A full flight-level ML model trained on real U.S. DOT/BTS on-time performance records: leakage-safe features, time-aware validation, candidate model selection and a tuned decision threshold.",
        "es": "Un modelo de ML completo a nivel de vuelo, entrenado con registros reales de puntualidad de U.S. DOT/BTS: variables sin fuga de datos, validación consciente del tiempo, selección de modelo candidato y un umbral de decisión ajustado.",
    },
    "whats_real_eu_title": {"en": "Europe path — transparent transfer demo", "es": "Ruta Europa — demo de transferencia transparente"},
    "whats_real_eu_copy": {
        "en": "The same core model, combined with a real aggregated UK CAA punctuality layer by route/airline/month. It is intentionally not presented as a Europe-calibrated flight-level model — that distinction stays visible in the UI instead of getting blurred.",
        "es": "El mismo modelo base, combinado con una capa real de puntualidad agregada de UK CAA por ruta/aerolínea/mes. Deliberadamente no se presenta como un modelo a nivel de vuelo calibrado para Europa: esa distinción se mantiene visible en la UI en lugar de difuminarse.",
    },

    "pipeline_title": {"en": "The pipeline, end to end", "es": "El pipeline, de punta a punta"},
    "stage1_title": {"en": "Real data", "es": "Datos reales"},
    "stage1_cap": {"en": "BTS + UK CAA sources", "es": "Fuentes BTS + UK CAA"},
    "stage2_title": {"en": "Clean & guard", "es": "Limpieza y control"},
    "stage2_cap": {"en": "Remove leakage columns", "es": "Elimina columnas con fuga"},
    "stage3_title": {"en": "Features", "es": "Variables"},
    "stage3_cap": {"en": "Train-only aggregates", "es": "Agregados solo de entrenamiento"},
    "stage4_title": {"en": "Select", "es": "Selección"},
    "stage4_cap": {"en": "Best model by PR-AUC", "es": "Mejor modelo por PR-AUC"},
    "stage5_title": {"en": "Tune & test", "es": "Ajuste y test"},
    "stage5_cap": {"en": "Threshold + held-out test", "es": "Umbral + test reservado"},
    "stage6_title": {"en": "Serve & watch", "es": "Servir y vigilar"},
    "stage6_cap": {"en": "API, UI, monitoring", "es": "API, UI, monitorización"},

    "decisions_title": {"en": "Engineering decisions worth explaining", "es": "Decisiones de ingeniería que vale la pena explicar"},
    "decision1_title": {"en": "Leakage-safe historical aggregates", "es": "Agregados históricos sin fuga de datos"},
    "decision1_copy": {
        "en": "Carrier/route/airport delay rates are strong features, but only safe if fit on the training split alone. FlightRisk splits first, fits every historical lookup on training rows only, then applies it to validation, test and live inference — with a global fallback for combinations it has rarely seen.",
        "es": "Las tasas de retraso por aerolínea, ruta o aeropuerto son variables potentes, pero solo son seguras si se ajustan únicamente con la partición de entrenamiento. FlightRisk primero divide los datos, ajusta cada tabla histórica solo con filas de entrenamiento, y luego la aplica a validación, test e inferencia en vivo, con un valor de respaldo global para combinaciones poco frecuentes.",
    },
    "decision2_title": {"en": "Validation-first model selection", "es": "Selección de modelo basada en validación"},
    "decision2_copy": {
        "en": "Candidate models are compared and the decision threshold is tuned on a validation split. Final metrics are reported once on a held-out test split, so the headline numbers aren't the same data used to pick the winner.",
        "es": "Los modelos candidatos se comparan y el umbral de decisión se ajusta con una partición de validación. Las métricas finales se reportan una sola vez sobre una partición de test reservada, de modo que los números finales no salen de los mismos datos usados para elegir al ganador.",
    },
    "decision3_title": {"en": "Time-aware, not random, splits", "es": "Particiones por tiempo, no aleatorias"},
    "decision3_copy": {
        "en": "Data is split chronologically whenever a flight date is available, mirroring the real task of predicting the future from the past instead of leaking tomorrow's patterns into training.",
        "es": "Los datos se dividen cronológicamente siempre que hay fecha de vuelo disponible, reflejando la tarea real de predecir el futuro a partir del pasado, en lugar de filtrar patrones del día siguiente hacia el entrenamiento.",
    },
    "decision4_title": {"en": "Why rank with Lift@10", "es": "Por qué priorizar con Lift@10"},
    "decision4_copy": {
        "en": "For a risk-ranking product, Lift@Top10% answers a sharper question than accuracy: does the flagged highest-risk decile actually concentrate a disproportionate share of real delays?",
        "es": "Para un producto de priorización de riesgo, Lift@Top10% responde una pregunta más precisa que la accuracy: ¿el decil marcado como de mayor riesgo concentra realmente una proporción desproporcionada de los retrasos reales?",
    },

    "tech_stack_title": {"en": "Tech stack", "es": "Stack tecnológico"},

    "limitations_title": {"en": "Known limitations & what's next", "es": "Limitaciones conocidas y próximos pasos"},
    "limitations_intro": {
        "en": "Documented on purpose — these are exactly the questions a technical interviewer should ask.",
        "es": "Documentadas a propósito: son justo las preguntas que debería hacer un entrevistador técnico.",
    },
    "lim1_title": {"en": "Training window", "es": "Ventana de entrenamiento"},
    "lim1_copy": {
        "en": "A single BTS month is fast to reproduce but doesn't capture full seasonality.",
        "es": "Un solo mes de BTS es rápido de reproducir, pero no capta toda la estacionalidad.",
    },
    "lim1_fix": {
        "en": "Mitigation: the pipeline accepts multiple monthly CSVs; `run_temporal_backtest` evaluates expanding windows.",
        "es": "Mitigación: el pipeline acepta varios CSV mensuales; `run_temporal_backtest` evalúa ventanas expansivas.",
    },
    "lim2_title": {"en": "Fixed default hyperparameters", "es": "Hiperparámetros fijos por defecto"},
    "lim2_copy": {
        "en": "The main run favors fast, reproducible candidates over exhaustive tuning.",
        "es": "La ejecución principal favorece candidatos rápidos y reproducibles frente a un ajuste exhaustivo.",
    },
    "lim2_fix": {
        "en": "Mitigation: `tune_hyperparameters` runs a time-aware randomized search on top.",
        "es": "Mitigación: `tune_hyperparameters` ejecuta una búsqueda aleatoria consciente del tiempo.",
    },
    "lim3_title": {"en": "Single-split evaluation", "es": "Evaluación con una sola partición"},
    "lim3_copy": {
        "en": "One time-aware split is appropriate for temporal data, but it's still just one split.",
        "es": "Una sola partición consciente del tiempo es adecuada para datos temporales, pero sigue siendo solo una partición.",
    },
    "lim3_fix": {
        "en": "Mitigation: bootstrap confidence intervals and expanding-window backtests quantify that uncertainty.",
        "es": "Mitigación: los intervalos de confianza por bootstrap y los backtests de ventana expansiva cuantifican esa incertidumbre.",
    },
    "lim4_title": {"en": "Europe is a different granularity", "es": "Europa tiene otra granularidad"},
    "lim4_copy": {
        "en": "It models route/airline punctuality patterns from aggregate data, not individual flight probability.",
        "es": "Modela patrones de puntualidad por ruta/aerolínea a partir de datos agregados, no la probabilidad de un vuelo individual.",
    },
    "lim4_fix": {
        "en": "Mitigation: the UI and docs keep the U.S. and European paths clearly separated instead of blending them.",
        "es": "Mitigación: la UI y la documentación mantienen las rutas de EE. UU. y Europa claramente separadas en lugar de mezclarlas.",
    },
    "lim5_title": {"en": "No hosted public endpoint", "es": "Sin endpoint público desplegado"},
    "lim5_copy": {
        "en": "Docker, CI and an AWS ECS/Fargate example exist, but this release doesn't ship a live hosted service.",
        "es": "Existen Docker, CI y un ejemplo de AWS ECS/Fargate, pero esta versión no despliega un servicio alojado en vivo.",
    },
    "lim5_fix": {
        "en": "Next step: a hosted FastAPI + Streamlit service with persistent monitoring storage.",
        "es": "Siguiente paso: un servicio alojado de FastAPI + Streamlit con almacenamiento de monitorización persistente.",
    },

    "disclaimer_title": {"en": "Scope", "es": "Alcance"},
    "disclaimer_copy": {
        "en": "Educational ML engineering portfolio project — not operational aviation, safety, or travel-booking advice.",
        "es": "Proyecto educativo de portfolio en ingeniería de ML: no es un consejo operativo de aviación, de seguridad ni para reservar viajes.",
    },
    "about_cta": {
        "en": "Try it yourself in the <b>Prediction cockpit</b> tab, or explore the live numbers in <b>Model intelligence</b>.",
        "es": "Pruébalo tú mismo en la pestaña <b>Cabina de predicción</b>, o explora los números en vivo en <b>Inteligencia del modelo</b>.",
    },
}


# ---------------------------------------------------------------------------
# Fixed English phrases generated by the backend (app/services + src/models)
# rather than authored in this UI layer. These are looked up by exact string
# match with a safe fallback to the original phrase, so a wording change in
# the backend never raises -- it just shows up untranslated until this table
# is updated.
# ---------------------------------------------------------------------------
BACKEND_STRING_TRANSLATIONS: dict[str, dict[str, str]] = {
    "evening/night scheduled departure": {"es": "salida programada de tarde o noche"},
    "weekend travel": {"es": "viaje en fin de semana"},
    "route historical delay rate": {"es": "tasa histórica de retraso de la ruta"},
    "carrier historical delay rate": {"es": "tasa histórica de retraso de la aerolínea"},
    "origin airport historical delay rate": {"es": "tasa histórica de retraso del aeropuerto de origen"},
    "destination airport historical delay rate": {"es": "tasa histórica de retraso del aeropuerto de destino"},
    "long scheduled flight duration": {"es": "duración de vuelo programada larga"},
    "no strong risk drivers identified; near-average flight profile": {
        "es": "no se identificaron factores de riesgo fuertes; perfil de vuelo cercano al promedio",
    },
    "European route context: elevated historical delay share": {
        "es": "Contexto de ruta europea: proporción histórica de retraso elevada",
    },
    "European route context: comparatively punctual route": {
        "es": "Contexto de ruta europea: ruta comparativamente puntual",
    },
    "European route context: near-average punctuality": {
        "es": "Contexto de ruta europea: puntualidad cercana al promedio",
    },
    (
        "European mode combines the BTS-trained flight-level model with an "
        "aggregated European punctuality context layer. Treat it as a portfolio "
        "transfer demo, not a Europe-calibrated operational model."
    ): {
        "es": (
            "El modo Europa combina el modelo a nivel de vuelo entrenado con datos BTS "
            "con una capa de contexto agregado de puntualidad europea. Trátalo como una "
            "demo de transferencia para portfolio, no como un modelo operativo calibrado "
            "para Europa."
        ),
    },
}


def translator(lang: str):
    """Return a `t(key, **kwargs) -> str` function bound to `lang`.

    Falls back to English, and finally to the raw key, so a missing
    translation never raises -- it just becomes visibly obvious (the key
    itself shows up in the UI) instead of crashing the page.
    """
    resolved_lang = lang if lang in SUPPORTED_LANGS else DEFAULT_LANG

    def t(key: str, **kwargs) -> str:
        entry = TRANSLATIONS.get(key)
        if entry is None:
            return key
        template = entry.get(resolved_lang) or entry.get(DEFAULT_LANG) or key
        return template.format(**kwargs) if kwargs else template

    return t


def translate_backend_string(phrase: str, lang: str) -> str:
    """Translate a fixed English phrase produced by the backend/service layer.

    Unknown phrases (e.g. if the backend copy changes) are returned unchanged
    rather than raising, so new/edited factor strings degrade gracefully to
    English instead of breaking the page.
    """
    if lang == "en":
        return phrase
    entry = BACKEND_STRING_TRANSLATIONS.get(phrase)
    if entry is None:
        return phrase
    return entry.get(lang, phrase)
