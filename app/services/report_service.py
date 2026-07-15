"""Generate bilingual Flight Delay Risk PDF briefs for review and export."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#164A73")
AMBER = colors.HexColor("#BB7A24")
IVORY = colors.HexColor("#F4F9FF")
LINE = colors.HexColor("#D8E7F5")
MUTED = colors.HexColor("#5D6F83")
INK = colors.HexColor("#10243E")

COPY = {
    "en": {
        "single_title": "Flight Delay Risk - Flight Risk Brief",
        "schedule_title": "Flight Delay Risk - Ranked Schedule Brief",
        "prepared": "Prepared for human review",
        "flight": "Flight",
        "route": "Route",
        "schedule": "Schedule",
        "probability": "Estimated chance of a 15+ minute arrival delay",
        "raw_score": "Uncalibrated model score",
        "route_rate": "Usual delay rate on this route",
        "relative": "Risk compared with this route",
        "support": "Earlier route flights behind the baseline",
        "coverage": "Historical route evidence",
        "seen": "Seen in training",
        "unseen": "Fallback used",
        "decision": "Review recommendation",
        "explanation": "What influenced this estimate",
        "increase": "Pushes risk up",
        "decrease": "Pushes risk down",
        "limitations": "Important limitations",
        "limitations_body": (
            "Schedule-only estimate. The model does not use live weather, aircraft rotation, ATC state, "
            "crew status or post-departure information. Contributions explain model behaviour, not real-world causes. "
            "Not for dispatch, safety or passenger guarantees."
        ),
        "summary": "Schedule summary",
        "flights": "Flights ranked",
        "priority": "Priority queue",
        "watch": "Watch queue",
        "average": "Average estimated delay risk",
        "top": "Highest estimated delay risk",
        "table_rank": "Rank",
        "table_flight": "Flight",
        "table_route": "Route",
        "table_departure": "Departure",
        "table_probability": "Probability",
        "table_cohort": "Usual route risk",
        "table_exposure": "Risk vs route",
        "table_support": "Earlier flights",
        "table_queue": "Queue",
        "footer": "Flight Delay Risk v1.5.0 - Built by Oriol Martínez - Portfolio ML system",
        "queue_priority": "Priority",
        "queue_watch": "Watch",
        "queue_routine": "Routine",
        "interpretation_above": "The estimate is {delta}% above the route baseline, based on {support} earlier route flights.",
        "interpretation_below": "The estimate is {delta}% below the route baseline, based on {support} earlier route flights.",
        "interpretation_equal": "The estimate is close to the route baseline, based on {support} earlier route flights.",
        "interpretation_title": "What this means",
        "contribution_note": "These are model associations, not proven real-world causes.",
        "schedule_note": "Priority and watch groups are relative to this uploaded schedule; they do not guarantee a delay.",
    },
    "es": {
        "single_title": "Flight Delay Risk - Informe de riesgo del vuelo",
        "schedule_title": "Flight Delay Risk - Informe del horario priorizado",
        "prepared": "Preparado para revisión humana",
        "flight": "Vuelo",
        "route": "Ruta",
        "schedule": "Horario",
        "probability": "Probabilidad estimada de llegar con 15+ minutos de retraso",
        "raw_score": "Score del modelo sin calibrar",
        "route_rate": "Riesgo habitual de retraso en esta ruta",
        "relative": "Riesgo comparado con esta ruta",
        "support": "Vuelos anteriores detrás de la referencia",
        "coverage": "Evidencia histórica de la ruta",
        "seen": "Visto en entrenamiento",
        "unseen": "Se usa fallback",
        "decision": "Recomendación de revisión",
        "explanation": "Qué influyó en esta estimación",
        "increase": "Eleva el riesgo",
        "decrease": "Reduce el riesgo",
        "limitations": "Limitaciones importantes",
        "limitations_body": (
            "Estimación basada solo en el horario. El modelo no utiliza meteorología en vivo, rotación de aeronave, "
            "estado ATC, tripulación ni información posterior a la salida. Las contribuciones explican el comportamiento "
            "del modelo, no causas reales. No usar para despacho, seguridad ni garantías al pasajero."
        ),
        "summary": "Resumen del horario",
        "flights": "Vuelos priorizados",
        "priority": "Cola prioritaria",
        "watch": "Cola de vigilancia",
        "average": "Riesgo estimado medio",
        "top": "Mayor riesgo estimado",
        "table_rank": "Pos.",
        "table_flight": "Vuelo",
        "table_route": "Ruta",
        "table_departure": "Salida",
        "table_probability": "Probabilidad",
        "table_cohort": "Riesgo habitual ruta",
        "table_exposure": "Riesgo vs ruta",
        "table_support": "Vuelos anteriores",
        "table_queue": "Cola",
        "footer": "Flight Delay Risk v1.5.0 - Creado por Oriol Martínez - Sistema ML de portfolio",
        "queue_priority": "Prioridad",
        "queue_watch": "Vigilancia",
        "queue_routine": "Rutina",
        "interpretation_above": "La estimación está un {delta}% por encima de la referencia de la ruta, basada en {support} vuelos anteriores.",
        "interpretation_below": "La estimación está un {delta}% por debajo de la referencia de la ruta, basada en {support} vuelos anteriores.",
        "interpretation_equal": "La estimación está cerca de la referencia de la ruta, basada en {support} vuelos anteriores.",
        "interpretation_title": "Qué significa",
        "contribution_note": "Son asociaciones del modelo, no causas reales demostradas.",
        "schedule_note": "Los grupos de prioridad y vigilancia son relativos al horario subido; no garantizan un retraso.",
    },
}

FEATURE_LABELS = {
    "en": {
        "RouteDelayRate": "Route historical rate",
        "CarrierRouteDelayRate": "Carrier-route historical rate",
        "OriginHourDelayRate": "Origin-hour historical rate",
        "DestHourDelayRate": "Destination-hour historical rate",
        "CarrierDelayRate": "Carrier historical rate",
        "OriginDelayRate": "Origin historical rate",
        "DestDelayRate": "Destination historical rate",
        "DepHour": "Scheduled departure hour",
        "ArrHour": "Scheduled arrival hour",
        "Distance": "Distance",
        "LogDistance": "Log distance",
        "CRSElapsedTime": "Scheduled duration",
        "Airline": "Carrier",
        "Origin": "Origin",
        "Dest": "Destination",
        "Route": "Route",
        "DepPeriod": "Departure period",
        "ArrPeriod": "Arrival period",
        "Month": "Month",
        "DayOfWeek": "Day of week",
    },
    "es": {
        "RouteDelayRate": "Tasa histórica de la ruta",
        "CarrierRouteDelayRate": "Tasa histórica aerolínea-ruta",
        "OriginHourDelayRate": "Tasa histórica origen-hora",
        "DestHourDelayRate": "Tasa histórica destino-hora",
        "CarrierDelayRate": "Tasa histórica de la aerolínea",
        "OriginDelayRate": "Tasa histórica del origen",
        "DestDelayRate": "Tasa histórica del destino",
        "DepHour": "Hora programada de salida",
        "ArrHour": "Hora programada de llegada",
        "Distance": "Distancia",
        "LogDistance": "Logaritmo de distancia",
        "CRSElapsedTime": "Duración programada",
        "Airline": "Aerolínea",
        "Origin": "Origen",
        "Dest": "Destino",
        "Route": "Ruta",
        "DepPeriod": "Franja de salida",
        "ArrPeriod": "Franja de llegada",
        "Month": "Mes",
        "DayOfWeek": "Día de la semana",
    },
}


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FRTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "FRSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
        ),
        "heading": ParagraphStyle(
            "FRHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "FRBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "FRSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
    }


def _footer(canvas: Any, doc: Any, text: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, doc.pagesize[0] - 18 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, text)
    canvas.drawRightString(doc.pagesize[0] - 18 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def _metric_table(rows: list[tuple[str, str]], width: float) -> Table:
    data = [[Paragraph(label, _styles()["small"]), Paragraph(value, _styles()["body"])] for label, value in rows]
    table = Table(data, colWidths=[width * 0.58, width * 0.42])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), IVORY),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_flight_brief_pdf(
    flight: dict[str, Any],
    prediction: dict[str, Any],
    context: dict[str, Any],
    *,
    lang: str = "en",
) -> bytes:
    """Build a compact bilingual single-flight PDF brief."""
    lang = "es" if lang == "es" else "en"
    t = COPY[lang]
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title=t["single_title"],
        author="Oriol Martínez",
    )

    probability = float(prediction.get("delay_probability", 0.0))
    route_rate = float(context.get("route_rate", 0.0))
    relative = probability / route_rate if route_rate > 0 else 0.0
    route = f"{flight.get('origin', '')} - {flight.get('destination', '')}"
    flight_id = " ".join(
        part for part in [str(flight.get("airline", "")), str(flight.get("flight_number", "")).strip()] if part
    )
    departure = str(flight.get("scheduled_departure", flight.get("crs_dep_time", "n/a")))
    arrival = str(flight.get("scheduled_arrival", flight.get("crs_arr_time", "n/a")))
    coverage = t["seen"] if context.get("route_seen") else t["unseen"]

    story: list[Any] = [
        Paragraph(t["single_title"], styles["title"]),
        Paragraph(t["prepared"], styles["subtitle"]),
        Spacer(1, 7 * mm),
        _metric_table(
            [
                (t["flight"], flight_id or "n/a"),
                (t["route"], route),
                (t["schedule"], f"{departure} - {arrival}"),
                (t["probability"], _pct(probability)),
                (t["route_rate"], _pct(route_rate)),
                (t["relative"], f"{relative:.2f}x"),
                (t["support"], f"{int(context.get('route_support', 0)):,}"),
                (t["coverage"], coverage),
                (
                    t["decision"],
                    t.get(
                        f"queue_{str(flight.get('review_label', prediction.get('risk_level', 'watch'))).lower()}",
                        str(flight.get("review_label", prediction.get("risk_level", "watch"))).title(),
                    ),
                ),
            ],
            A4[0] - 36 * mm,
        ),
        Spacer(1, 4 * mm),
    ]

    support_count = int(context.get("route_support", 0))
    if relative > 1.02:
        interpretation = t["interpretation_above"].format(delta=round((relative - 1) * 100), support=f"{support_count:,}")
    elif relative < 0.98:
        interpretation = t["interpretation_below"].format(delta=round((1 - relative) * 100), support=f"{support_count:,}")
    else:
        interpretation = t["interpretation_equal"].format(support=f"{support_count:,}")
    story.extend([
        Paragraph(t["interpretation_title"], styles["heading"]),
        Paragraph(interpretation, styles["body"]),
        Spacer(1, 3 * mm),
        Paragraph(t["explanation"], styles["heading"]),
        Paragraph(t["contribution_note"], styles["small"]),
        Spacer(1, 2 * mm),
    ])

    contributions = prediction.get("local_contributions") or []
    if contributions:
        explanation_rows = [[Paragraph(t["explanation"], styles["small"]), Paragraph("Effect" if lang == "en" else "Efecto", styles["small"])]]
        for item in contributions[:6]:
            label = FEATURE_LABELS[lang].get(str(item.get("feature")), str(item.get("feature")))
            value = item.get("active_category") or item.get("raw_value")
            value_text = "" if value is None else str(value)
            contribution = float(item.get("contribution", 0.0))
            effect = t["increase"] if contribution >= 0 else t["decrease"]
            explanation_rows.append([
                Paragraph(f"{label}<br/><font color='#5D6F83'>{value_text}</font>", styles["body"]),
                Paragraph(effect, styles["body"]),
            ])
        explanation_table = Table(explanation_rows, colWidths=[125 * mm, 46 * mm])
        explanation_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), IVORY),
            ("BOX", (0, 0), (-1, -1), 0.5, LINE),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica-Bold"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(explanation_table)
    else:
        story.append(Paragraph("No explanation is available for this artifact." if lang == "en" else "No hay una explicación disponible para este artefacto.", styles["body"]))

    story.extend(
        [
            Spacer(1, 4 * mm),
            KeepTogether(
                [
                    Paragraph(t["limitations"], styles["heading"]),
                    Paragraph(t["limitations_body"], styles["body"]),
                ]
            ),
        ]
    )

    doc.build(story, onFirstPage=lambda c, d: _footer(c, d, t["footer"]), onLaterPages=lambda c, d: _footer(c, d, t["footer"]))
    return buffer.getvalue()


def build_schedule_brief_pdf(ranked: pd.DataFrame, *, lang: str = "en") -> bytes:
    """Build a landscape PDF summary for an uploaded ranked schedule."""
    lang = "es" if lang == "es" else "en"
    t = COPY[lang]
    styles = _styles()
    buffer = BytesIO()
    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        title=t["schedule_title"],
        author="Oriol Martínez",
    )

    total = len(ranked)
    priority = int((ranked.get("priority_tier") == "Priority").sum()) if total else 0
    watch = int((ranked.get("priority_tier") == "Watch").sum()) if total else 0
    average = float(ranked["delay_probability"].mean()) if total else 0.0
    highest = float(ranked["delay_probability"].max()) if total else 0.0

    story: list[Any] = [
        Paragraph(t["schedule_title"], styles["title"]),
        Paragraph(t["prepared"], styles["subtitle"]),
        Spacer(1, 5 * mm),
        Paragraph(t["summary"], styles["heading"]),
    ]

    summary_data = [
        [
            Paragraph(t["flights"], styles["small"]),
            Paragraph(t["priority"], styles["small"]),
            Paragraph(t["watch"], styles["small"]),
            Paragraph(t["average"], styles["small"]),
            Paragraph(t["top"], styles["small"]),
        ],
        [
            Paragraph(str(total), styles["body"]),
            Paragraph(str(priority), styles["body"]),
            Paragraph(str(watch), styles["body"]),
            Paragraph(_pct(average), styles["body"]),
            Paragraph(_pct(highest), styles["body"]),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[(pagesize[0] - 28 * mm) / 5] * 5)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), IVORY),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 2 * mm), Paragraph(t["schedule_note"], styles["small"]), Spacer(1, 4 * mm)])

    headers = [
        t["table_rank"],
        t["table_flight"],
        t["table_route"],
        t["table_departure"],
        t["table_probability"],
        t["table_cohort"],
        t["table_exposure"],
        t["table_support"],
        t["table_queue"],
    ]
    rows: list[list[Any]] = [headers]
    for _, row in ranked.head(40).iterrows():
        flight_id = " ".join(
            part for part in [str(row.get("airline", "")), str(row.get("flight_number", "")).strip()] if part and part != "nan"
        )
        route = f"{row.get('origin', '')} - {row.get('destination', '')}"
        rows.append(
            [
                int(row.get("rank", 0)),
                flight_id or "-",
                route,
                str(row.get("scheduled_departure", row.get("crs_dep_time", "-"))),
                _pct(row.get("delay_probability")),
                _pct(row.get("route_rate")),
                f"{float(row.get('relative_exposure', 0.0)):.2f}x",
                f"{int(row.get('route_support', 0)):,}",
                t.get(f"queue_{str(row.get('priority_tier', 'Routine')).lower()}", str(row.get("priority_tier", "Routine"))),
            ]
        )

    table = Table(
        rows,
        repeatRows=1,
        colWidths=[14 * mm, 24 * mm, 31 * mm, 25 * mm, 28 * mm, 23 * mm, 22 * mm, 20 * mm, 25 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IVORY]),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (4, 1), (7, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 5 * mm),
            Paragraph(t["limitations"], styles["heading"]),
            Paragraph(t["limitations_body"], styles["body"]),
        ]
    )

    footer = t["footer"]
    doc.build(story, onFirstPage=lambda c, d: _footer(c, d, footer), onLaterPages=lambda c, d: _footer(c, d, footer))
    return buffer.getvalue()
