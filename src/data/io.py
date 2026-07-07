"""Small I/O helpers for processed datasets.

The preferred processed format is Parquet. For lightweight CI/sandbox
runs where an optional Parquet engine is not installed, the helpers fall
back to CSV while keeping the public CLI commands stable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _csv_fallback_path(path: Path) -> Path:
    """Return a sibling CSV path for a parquet path, or the path itself if already CSV."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return path
    return path.with_suffix(".csv")


def write_processed_frame(df: pd.DataFrame, path: Path) -> Path:
    """Write a processed dataset.

    Parquet is preferred because BTS monthly files are large. If pyarrow or
    fastparquet is unavailable, write a CSV sibling and return that actual path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path

    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError as exc:
        fallback = _csv_fallback_path(path)
        logger.warning(
            "Could not write Parquet because an optional engine is missing (%s). "
            "Writing CSV fallback instead: %s",
            exc,
            fallback,
        )
        df.to_csv(fallback, index=False)
        return fallback


def read_processed_frame(path: Path) -> pd.DataFrame:
    """Read a processed dataset from Parquet or CSV.

    If the requested Parquet path does not exist but a CSV fallback sibling does,
    the CSV is loaded automatically. This keeps commands ergonomic on machines
    without a Parquet engine installed.
    """
    path = Path(path)

    if path.exists() and path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.exists():
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            fallback = _csv_fallback_path(path)
            if fallback.exists():
                logger.warning(
                    "Could not read Parquet because an optional engine is missing (%s). "
                    "Reading CSV fallback instead: %s",
                    exc,
                    fallback,
                )
                return pd.read_csv(fallback)
            raise

    fallback = _csv_fallback_path(path)
    if fallback.exists():
        logger.warning("Processed path %s not found; reading CSV fallback %s", path, fallback)
        return pd.read_csv(fallback)

    raise FileNotFoundError(f"Processed dataset not found: {path} or fallback {fallback}")
