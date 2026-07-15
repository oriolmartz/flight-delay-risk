"""Run the canonical FlightRisk training protocol on bundled sample data."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.train_model import TrainingConfig, run_training
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SAMPLE_CSV_PATH
from src.data.clean import clean_flights
from src.data.io import write_processed_frame
from src.data.load_data import normalize_columns
from src.data.manifest import sha256_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== FlightRisk local demo: canonical temporal workflow ===")
    raw_df = normalize_columns(pd.read_csv(SAMPLE_CSV_PATH))
    clean_df = clean_flights(raw_df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = write_processed_frame(
        clean_df, PROCESSED_DATA_DIR / "flights_processed_demo.parquet"
    )
    manifest_path = PROCESSED_DATA_DIR / "data_manifest_demo.json"
    manifest = {
        "manifest_version": "demo-1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SAMPLE_CSV_PATH),
        "output_path": str(Path(processed_path).resolve()),
        "output_sha256": sha256_file(Path(processed_path)),
        "dataset": {
            "rows": len(clean_df),
            "date_start": str(pd.to_datetime(clean_df["FlightDate"]).min().date()),
            "date_end": str(pd.to_datetime(clean_df["FlightDate"]).max().date()),
            "target_positive_rate": float(clean_df["ArrDel15"].mean()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = run_training(
        TrainingConfig(
            data=Path(processed_path),
            output=MODELS_DIR / "flightrisk_model.joblib",
            data_manifest=manifest_path,
            candidate_profile="full",
            bootstrap_samples=0,
        )
    )
    logger.info("Demo complete: %s", json.dumps(result, indent=2))
    logger.info("API: python -m uvicorn app.api.main:app --reload --port 8000")
    logger.info("UI:  python -m streamlit run app/dashboard/streamlit_app.py")


if __name__ == "__main__":
    main()
