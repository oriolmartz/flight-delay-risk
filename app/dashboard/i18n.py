"""Small bilingual copy layer for the FlightRisk product UI."""

TEXT = {
    "en": {
        "language": "English",
        "tabs": ["Analyze flight", "Rank schedule", "Validation", "Model & operations"],
        "hero_kicker": "Pre-departure risk · schedule triage",
        "hero_title": "Know which flights deserve <em>attention before departure.</em>",
        "hero_sub": (
            "FlightRisk ranks scheduled flights by estimated arrival-delay exposure using only "
            "information available before take-off. It surfaces historical context, data support "
            "and the limits of the current model artifact."
        ),
        "constraint": (
            "Schedule-only estimate · no live weather, aircraft rotation or ATC state · "
            "not for operational dispatch or passenger guarantees."
        ),
        "analyze_title": "Analyze one scheduled flight",
        "analyze_sub": "Enter natural schedule fields. Calendar features are derived automatically.",
        "rank_title": "Rank a flight schedule",
        "rank_sub": "Upload or load a schedule and turn model output into a review queue.",
        "validation_title": "Validation evidence",
        "validation_sub": "Real committed results, including model comparison and calibration limitations.",
        "operations_title": "Model and operations",
        "operations_sub": "Artifact lineage, leakage contract, API surface and production boundaries.",
    },
    "es": {
        "language": "Español",
        "tabs": ["Analizar vuelo", "Priorizar horario", "Validación", "Modelo y operaciones"],
        "hero_kicker": "Riesgo antes de salida · priorización de horarios",
        "hero_title": "Detecta qué vuelos merecen <em>atención antes de despegar.</em>",
        "hero_sub": (
            "FlightRisk ordena vuelos programados por exposición estimada a retraso usando solo "
            "información disponible antes de la salida. Muestra contexto histórico, soporte de datos "
            "y los límites del artefacto actual."
        ),
        "constraint": (
            "Estimación basada en horario · sin meteorología en vivo, rotación de aeronave ni estado ATC · "
            "no usar para despacho operacional ni garantías al pasajero."
        ),
        "analyze_title": "Analizar un vuelo programado",
        "analyze_sub": "Introduce campos naturales del horario. Las variables de calendario se derivan automáticamente.",
        "rank_title": "Priorizar un horario de vuelos",
        "rank_sub": "Sube o carga un horario y convierte la salida del modelo en una cola de revisión.",
        "validation_title": "Evidencia de validación",
        "validation_sub": "Resultados reales incluidos en el repositorio, con comparación y límites de calibración.",
        "operations_title": "Modelo y operaciones",
        "operations_sub": "Linaje del artefacto, contrato anti-leakage, API y límites de producción.",
    },
}
