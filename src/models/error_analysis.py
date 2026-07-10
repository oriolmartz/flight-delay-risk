"""Error analysis utilities for FlightRisk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import REPORTS_DIR


def _bucket_distance(distance: float) -> str:
    if distance < 500:
        return "short_<500mi"
    if distance < 1500:
        return "medium_500_1499mi"
    return "long_1500mi_plus"


def _bucket_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def subgroup_metrics(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(group_col, dropna=False):
        if len(group) < 5:
            continue
        rows.append(
            {
                "segment": str(value),
                "n": int(len(group)),
                "actual_delay_rate": float(group["y_true"].mean()),
                "predicted_delay_rate": float(group["y_pred"].mean()),
                "precision": float(precision_score(group["y_true"], group["y_pred"], zero_division=0)),
                "recall": float(recall_score(group["y_true"], group["y_pred"], zero_division=0)),
                "f1": float(f1_score(group["y_true"], group["y_pred"], zero_division=0)),
            }
        )
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows[:20]


def build_error_analysis(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_proba,
    threshold: float,
) -> dict[str, Any]:
    df = X_test.copy().reset_index(drop=True)
    df["y_true"] = y_true.reset_index(drop=True).astype(int)
    df["y_proba"] = y_proba
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)
    df["distance_band"] = df["Distance"].apply(_bucket_distance)
    df["departure_period"] = df["DepHour"].apply(_bucket_hour)

    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]

    return {
        "threshold": float(threshold),
        "rows": int(len(df)),
        "false_positives": int(len(fp)),
        "false_negatives": int(len(fn)),
        "false_positive_rate_in_predictions": float(len(fp) / max(1, int(df["y_pred"].sum()))),
        "false_negative_rate_in_actual_delays": float(len(fn) / max(1, int(df["y_true"].sum()))),
        "segments": {
            "airline": subgroup_metrics(df, "Airline"),
            "origin": subgroup_metrics(df, "Origin"),
            "destination": subgroup_metrics(df, "Dest"),
            "departure_period": subgroup_metrics(df, "departure_period"),
            "distance_band": subgroup_metrics(df, "distance_band"),
        },
    }


def save_error_analysis(analysis: dict[str, Any], out_dir: Path = REPORTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(json.dumps(analysis, indent=2))

    lines = ["# FlightRisk Error Analysis", ""]
    lines.append(f"Rows: {analysis['rows']}")
    lines.append(f"Threshold: {analysis['threshold']:.3f}")
    lines.append(f"False positives: {analysis['false_positives']}")
    lines.append(f"False negatives: {analysis['false_negatives']}")
    lines.append("")
    for name, rows in analysis["segments"].items():
        lines.append(f"## Segment: {name}")
        lines.append("segment | n | actual_delay_rate | predicted_delay_rate | precision | recall | f1")
        lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---:")
        for r in rows[:10]:
            lines.append(
                f"{r['segment']} | {r['n']} | {r['actual_delay_rate']:.3f} | {r['predicted_delay_rate']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f}"
            )
        lines.append("")
    (out_dir / "error_analysis.md").write_text("\n".join(lines))
