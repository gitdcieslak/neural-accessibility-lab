from __future__ import annotations

import matplotlib.pyplot as plt

from neural_accessibility_lab.analyses.dashboard import plot_drift_dashboard


def plot_metric(result_or_frame, metric: str, ax=None, **plot_kwargs):
    """Plot one metric over epochs from an ExperimentResult or DataFrame."""

    df = result_or_frame.to_frame() if hasattr(result_or_frame, "to_frame") else result_or_frame
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df[metric], marker="o", **plot_kwargs)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    return ax


__all__ = ["plot_metric", "plot_drift_dashboard"]
