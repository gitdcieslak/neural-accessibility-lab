from __future__ import annotations

import numpy as np
import pandas as pd


def centroid_drift(previous_features: np.ndarray, current_features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute class-centroid movement between two feature snapshots."""

    prev = np.asarray(previous_features, dtype=float)
    curr = np.asarray(current_features, dtype=float)
    y = np.asarray(labels)
    out: dict[str, float] = {}
    drifts = []

    for cls in np.unique(y):
        mask = y == cls
        if not mask.any():
            continue
        value = float(np.linalg.norm(curr[mask].mean(axis=0) - prev[mask].mean(axis=0)))
        out[f"centroid_drift_{cls}"] = value
        drifts.append(value)

    out["centroid_drift_mean"] = float(np.mean(drifts)) if drifts else np.nan
    return out


def add_delta_columns(df: pd.DataFrame, columns: list[str] | None = None, groupby: str | None = None) -> pd.DataFrame:
    """Return a copy with per-row deltas for metric columns."""

    out = df.copy()
    if columns is None:
        columns = [c for c in out.select_dtypes(include="number").columns if c != "epoch"]

    grouped = out.groupby(groupby, sort=False) if groupby else [(None, out)]
    for column in columns:
        delta = pd.Series(index=out.index, dtype="float64")
        for _, part in grouped:
            delta.loc[part.index] = part[column].diff()
        out[f"delta_{column}"] = delta
    return out


def normalize(values: np.ndarray | pd.Series) -> np.ndarray:
    """Min-max normalize values, preserving constant arrays as zeros."""

    x = np.asarray(values, dtype=float)
    low = np.nanmin(x)
    high = np.nanmax(x)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.zeros_like(x, dtype=float)
    return (x - low) / (high - low)
