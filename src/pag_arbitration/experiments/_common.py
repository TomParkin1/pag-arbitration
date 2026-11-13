"""Helpers shared by the experiment modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

THREAT_LEVELS = ["low", "medium", "high"]


def seeds(episodes: int, seed_base: int) -> range:
    """Seeds shared by every condition, so controllers face identical worlds."""
    return range(seed_base, seed_base + episodes)


def write_csv(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    """Write ``df`` to ``out_dir/name`` (creating the directory) and report it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    return path
