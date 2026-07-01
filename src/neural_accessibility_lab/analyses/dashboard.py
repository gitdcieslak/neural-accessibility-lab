from __future__ import annotations

from neural_accessibility_lab.metrics import add_delta_columns


def build_drift_dashboard_frame(result):
    """Build an epoch metric frame with simple delta columns for dashboard analysis."""

    frame = result.to_frame() if hasattr(result, "to_frame") else result.copy()
    return add_delta_columns(frame)


def plot_drift_dashboard(result_or_frame, lr=None):
    """Plot core accessibility dynamics from an ExperimentResult or metric frame."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("plot_drift_dashboard requires matplotlib. Install with `pip install 'neural-accessibility-lab[ml]'`.") from exc

    frame = build_drift_dashboard_frame(result_or_frame) if hasattr(result_or_frame, "to_frame") else result_or_frame
    metrics = [m for m in ["val_auc", "survival_auc", "delta_survival_auc", "centroid_distance", "hidden_hellinger_mean"] if m in frame.columns]
    if not metrics:
        raise ValueError("No dashboard metrics found in frame.")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.plot(frame["epoch"], frame[metric], marker="o")
        ax.set_ylabel(metric)
        if lr is not None:
            ax.set_title(f"{metric} (lr={lr})")
        else:
            ax.set_title(metric)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    return fig, axes
