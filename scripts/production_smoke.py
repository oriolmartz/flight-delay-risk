"""Run the public deployment contract against the packaged FastAPI app."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

from app.api.main import app
from app.services import prediction_service
from src.version import APP_VERSION

SAMPLE = {
    "airline": "DL",
    "origin": "JFK",
    "destination": "LAX",
    "month": 7,
    "day_of_week": 5,
    "crs_dep_time": 1830,
    "crs_arr_time": 2145,
    "crs_elapsed_time": 375,
    "distance": 2475,
    "flight_date": "2024-07-12",
}


def run_smoke(output: Path) -> dict:
    prediction_service.get_artifact.cache_clear()
    client = TestClient(app)
    checks: dict[str, bool] = {}
    timings: dict[str, float] = {}

    def request(name: str, method: str, path: str, **kwargs):
        started = perf_counter()
        response = client.request(method, path, **kwargs)
        timings[name] = round((perf_counter() - started) * 1000, 3)
        checks[f"{name}_status"] = response.status_code == 200
        checks[f"{name}_version_header"] = response.headers.get("x-flightrisk-version") == APP_VERSION
        checks[f"{name}_request_id"] = bool(response.headers.get("x-request-id"))
        return response

    live = request("live", "GET", "/live")
    ready = request("ready", "GET", "/ready")
    info = request("model_info", "GET", "/model/info")
    predict = request("predict", "POST", "/predict", json=SAMPLE)
    rank = request("rank", "POST", "/rank", json={"flights": [SAMPLE, {**SAMPLE, "origin": "ATL", "destination": "BOS", "distance": 946, "crs_elapsed_time": 150}]})
    openapi = request("openapi", "GET", "/openapi.json")

    checks.update(
        {
            "live_payload": live.json().get("status") == "alive",
            "ready_payload": ready.json().get("status") == "ready",
            "artifact_schema_v7": int(info.json().get("artifact_schema_version", 0)) >= 7,
            "release_version_match": info.json().get("version") == APP_VERSION,
            "scaled_refit_rows": int(info.json().get("n_train_rows", 0) or 0) >= 150_000,
            "valid_probability": 0.0 <= float(predict.json().get("delay_probability", -1)) <= 1.0,
            "ranked_two_flights": rank.json().get("flights_ranked") == 2,
            "openapi_contract": "/ready" in openapi.json().get("paths", {}),
        }
    )
    payload = {
        "release": APP_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "timings_ms": timings,
        "model": {
            "name": info.json().get("model_name"),
            "trained_rows": info.json().get("n_train_rows"),
            "test_rows": info.json().get("n_test_rows"),
            "schema": info.json().get("artifact_schema_version"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Production smoke failed: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/production_smoke.json"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == "__main__":
    main()
