"""Memory-efficient deterministic release sampling from the canonical parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import RANDOM_SEED


def read_release_frame(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    table = pq.read_table(path)
    if max_rows is None or table.num_rows <= max_rows:
        frame = table.to_pandas()
    else:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        dates = pd.DatetimeIndex(table.column("FlightDate").to_numpy())
        month_codes = dates.year * 100 + dates.month
        unique, counts = np.unique(month_codes, return_counts=True)
        allocations = np.maximum(1, np.floor(counts / counts.sum() * max_rows).astype(int))
        while allocations.sum() > max_rows:
            eligible = np.where(allocations > 1)[0]
            allocations[eligible[np.argmax(allocations[eligible])]] -= 1
        while allocations.sum() < max_rows:
            allocations[np.argmax(counts - allocations)] += 1
        selected: list[np.ndarray] = []
        for offset, (month, allocation) in enumerate(zip(unique, allocations, strict=False)):
            positions = np.flatnonzero(month_codes == month)
            rng = np.random.default_rng(RANDOM_SEED + offset)
            selected.append(rng.choice(positions, size=min(int(allocation), len(positions)), replace=False))
        indices = np.sort(np.concatenate(selected))
        frame = table.take(pa.array(indices)).to_pandas()
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"], errors="raise", format="mixed")
    return frame.sort_values(["FlightDate", "CRSDepTime"], kind="stable").reset_index(drop=True)
