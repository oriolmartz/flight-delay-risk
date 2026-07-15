"""Source discovery, duplicate-month protection and dataset fingerprints."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data.load_data import normalize_columns


class DuplicateMonthError(ValueError):
    """Raised when multiple raw files claim the same BTS year/month."""


@dataclass
class SourceFileInfo:
    path: str
    name: str
    size_bytes: int
    sha256: str
    year: int
    month: int
    selected: bool = True
    selection_reason: str = "unique_month"

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source_file(path: Path) -> SourceFileInfo:
    path = Path(path)
    preview = normalize_columns(pd.read_csv(path, nrows=256, low_memory=False))
    missing = [column for column in ("Year", "Month") if column not in preview.columns]
    if missing:
        raise ValueError(f"Cannot identify source month for {path.name}; missing {missing}")
    years = pd.to_numeric(preview["Year"], errors="coerce").dropna().astype(int).unique()
    months = pd.to_numeric(preview["Month"], errors="coerce").dropna().astype(int).unique()
    if len(years) != 1 or len(months) != 1:
        raise ValueError(
            f"Expected one BTS year/month in {path.name}; found years={years.tolist()} "
            f"months={months.tolist()}"
        )
    return SourceFileInfo(
        path=str(path.resolve()),
        name=path.name,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        year=int(years[0]),
        month=int(months[0]),
    )


def resolve_monthly_sources(
    input_dir: Path,
    *,
    duplicate_month_policy: str = "error",
) -> tuple[list[Path], list[SourceFileInfo]]:
    """Resolve one source file per year/month and make any conflict explicit.

    Supported policies:
    - ``error``: fail before reading the dataset;
    - ``prefer-largest``: select the largest file in each duplicate group;
    - ``prefer-newest``: select the most recently modified file;
    - ``first``: select the lexicographically first filename.
    """
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    if duplicate_month_policy not in {"error", "prefer-largest", "prefer-newest", "first"}:
        raise ValueError(
            "duplicate_month_policy must be one of: error, prefer-largest, prefer-newest, first"
        )

    infos = [inspect_source_file(path) for path in paths]
    groups: dict[tuple[int, int], list[SourceFileInfo]] = {}
    for info in infos:
        groups.setdefault((info.year, info.month), []).append(info)

    selected: list[Path] = []
    conflicts = {key: values for key, values in groups.items() if len(values) > 1}
    if conflicts and duplicate_month_policy == "error":
        details = "; ".join(
            f"{year}-{month:02d}: {[item.name for item in values]}"
            for (year, month), values in sorted(conflicts.items())
        )
        raise DuplicateMonthError(
            "Multiple raw files represent the same BTS month. Remove/quarantine the duplicate "
            f"or choose an explicit policy. Conflicts: {details}"
        )

    for key, values in sorted(groups.items()):
        if len(values) == 1:
            chosen = values[0]
        elif duplicate_month_policy == "prefer-largest":
            chosen = max(values, key=lambda item: (item.size_bytes, item.name))
        elif duplicate_month_policy == "prefer-newest":
            chosen = max(values, key=lambda item: (Path(item.path).stat().st_mtime_ns, item.name))
        else:
            chosen = min(values, key=lambda item: item.name)

        for item in values:
            item.selected = item is chosen
            item.selection_reason = (
                "unique_month" if len(values) == 1 else f"duplicate_month_{duplicate_month_policy}"
            )
        selected.append(Path(chosen.path))

    return selected, infos
