"""Export the release OpenAPI contract for review and deployment checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.api.main import app
from src.version import APP_VERSION


def export_openapi(output: Path) -> dict:
    schema = app.openapi()
    schema.setdefault("info", {})["version"] = APP_VERSION
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return {"release": APP_VERSION, "output": str(output), "paths": len(schema.get("paths", {}))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/openapi.json"))
    args = parser.parse_args()
    print(json.dumps(export_openapi(args.output), indent=2))


if __name__ == "__main__":
    main()
