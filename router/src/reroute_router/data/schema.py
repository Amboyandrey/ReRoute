"""Unified schema every dataset builder (RouterBench, Arena, our own set)
normalizes into, so the eval harness and trainers never special-case the
data source.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["prompt_id", "prompt", "model", "quality", "cost_usd", "latency_ms", "source"]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def validate(df: pd.DataFrame) -> None:
    """Raises if df doesn't match the unified schema contract."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {sorted(missing)}")
    if not df["quality"].between(0.0, 1.0).all():
        bad = df.loc[~df["quality"].between(0.0, 1.0), "quality"]
        raise ValueError(f"quality must be in [0, 1], found: {bad.unique()[:5]}")
    if (df["cost_usd"] < 0).any():
        raise ValueError("cost_usd must be non-negative")
