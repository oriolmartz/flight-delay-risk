"""Download real UK CAA punctuality CSV files.

Usage:
    python -m scripts.download_uk_caa_punctuality --year 2024
    python -m scripts.download_uk_caa_punctuality --year 2024 --kind full-analysis

The script scrapes the official CAA year page and downloads CSV resources to
`data/europe/uk_caa_raw/`. Then run:

    python -m scripts.prepare_uk_caa_context

Official source page pattern:
https://www.caa.co.uk/data-and-analysis/uk-aviation-market/flight-punctuality/uk-flight-punctuality-statistics/<YEAR>/
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

CAA_YEAR_URL = "https://www.caa.co.uk/data-and-analysis/uk-aviation-market/flight-punctuality/uk-flight-punctuality-statistics/{year}/"


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def discover_caa_csv_links(year: int, kind: str = "full-analysis") -> list[dict]:
    url = CAA_YEAR_URL.format(year=year)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    html_text = response.text

    links: list[dict] = []
    pattern = re.compile(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>\s*(?P<label>.*?)\s*</a>', re.I | re.S)
    for match in pattern.finditer(html_text):
        label = re.sub(r"\s+", " ", html.unescape(re.sub("<.*?>", "", match.group("label")))).strip()
        href = html.unescape(match.group("href"))
        label_lower = label.lower()
        if "csv" not in label_lower or "punctuality statistics" not in label_lower:
            continue
        if kind == "full-analysis" and "full analysis" not in label_lower:
            continue
        if kind == "arrival-departure" and "arrival departure" not in label_lower:
            continue
        if kind == "summary" and "summary analysis" not in label_lower:
            continue
        if kind == "all":
            pass
        links.append({"label": label, "url": urljoin(url, href)})
    return links


def download_links(links: list[dict], output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for item in links:
        label = item["label"]
        url = item["url"]
        name = _slugify(label)
        if not name.endswith("csv"):
            name += ".csv"
        path = output_dir / name
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        downloaded.append({"label": label, "url": url, "path": str(path), "bytes": len(r.content)})
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real UK CAA punctuality CSV files.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--kind", choices=["full-analysis", "arrival-departure", "summary", "all"], default="full-analysis")
    parser.add_argument("--output-dir", default="data/europe/uk_caa_raw")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    links = discover_caa_csv_links(args.year, args.kind)
    if args.limit:
        links = links[: args.limit]
    result = download_links(links, args.output_dir)
    print(json.dumps({"year": args.year, "kind": args.kind, "downloaded": result}, indent=2))


if __name__ == "__main__":
    main()
