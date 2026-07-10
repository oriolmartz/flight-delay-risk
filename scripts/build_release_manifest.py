"""Build the FlightRisk release manifest with hashes for public artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.version import APP_VERSION, RELEASE_NAME

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "RELEASE_MANIFEST.json"
TRACKED = [
    "models/flightrisk_model.joblib",
    "reports/metrics.json",
    "reports/temporal_backtest.json",
    "reports/calibration_report.json",
    "reports/candidate_benchmark.json",
    "reports/performance_benchmark.json",
    "reports/ui_smoke.json",
    "README.md",
    "README_ES.md",
    "requirements.txt",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    files = {}
    for relative in TRACKED:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(relative)
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}

    manifest = {
        "name": "FlightRisk",
        "version": APP_VERSION,
        "release_name": RELEASE_NAME,
        "status": "stable_public_portfolio_release",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_schema_version": "2",
        "interfaces": ["streamlit_en_es", "fastapi", "pdf_reports_en_es"],
        "quality": {
            "temporal_backtest_folds": 4,
            "calibration": "isotonic",
            "historical_encoding": "strictly_prior_flight_date",
        },
        "files": files,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
