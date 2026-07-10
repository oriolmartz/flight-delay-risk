"""Run the bilingual Streamlit workflow with Streamlit's AppTest harness."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    app = AppTest.from_file(str(ROOT / "app" / "dashboard" / "streamlit_app.py"), default_timeout=40)
    app.run()
    if app.exception:
        raise RuntimeError(f"English Streamlit smoke failed: {app.exception}")
    app.selectbox(key="language_selector").set_value("Español").run()
    if app.exception:
        raise RuntimeError(f"Spanish Streamlit smoke failed: {app.exception}")
    buttons = [button for button in app.button if button.label == "Cargar horario de ejemplo"]
    if not buttons:
        raise RuntimeError("Spanish sample-schedule button is missing")
    buttons[0].click().run()
    if app.exception:
        raise RuntimeError(f"Spanish batch workflow failed: {app.exception}")
    download_labels = {item.label for item in app.get("download_button")}
    if "Descargar informe del horario (PDF)" not in download_labels:
        raise RuntimeError("Spanish schedule PDF export is missing")
    report = {
        "release": "1.0.0",
        "status": "passed",
        "languages": ["en", "es"],
        "sample_schedule": "passed",
        "pdf_export": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "reports" / "ui_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Bilingual Streamlit smoke passed")
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
