from __future__ import annotations

import numpy as np


def hellinger_1d(a: np.ndarray, b: np.ndarray, bins: int = 30, value_range: tuple[float, float] | None = None) -> float:
    """Hellinger distance between two one-dimensional empirical distributions."""

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan

    if value_range is None:
        low = float(np.nanmin([x.min(), y.min()]))
        high = float(np.nanmax([x.max(), y.max()]))
        if low == high:
            high = low + 1.0
        value_range = (low, high)

    px, _ = np.histogram(x, bins=bins, range=value_range, density=False)
    py, _ = np.histogram(y, bins=bins, range=value_range, density=False)
    px = px / max(px.sum(), 1)
    py = py / max(py.sum(), 1)
    return float(np.sqrt(np.sum((np.sqrt(px) - np.sqrt(py)) ** 2)) / np.sqrt(2))


def hidden_hellinger_metrics(features: np.ndarray, labels: np.ndarray, bins: int = 30) -> dict[str, float]:
    """Compute per-dimension class Hellinger distances for hidden features."""

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    classes = np.unique(y)
    if X.ndim != 2 or len(classes) < 2:
        return {"hidden_hellinger_mean": np.nan, "hidden_hellinger_max": np.nan}

    distances = [hellinger_1d(X[y == classes[0], i], X[y == classes[1], i], bins=bins) for i in range(X.shape[1])]
    return {"hidden_hellinger_mean": float(np.nanmean(distances)), "hidden_hellinger_max": float(np.nanmax(distances))}
