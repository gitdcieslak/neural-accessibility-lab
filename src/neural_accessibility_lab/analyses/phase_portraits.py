from __future__ import annotations


def plot_phase_portrait(frame, x, y, lr=None, title=None):
    """Plot one metric against another across epochs."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("plot_phase_portrait requires matplotlib. Install with `pip install 'neural-accessibility-lab[ml]'`.") from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(frame[x], frame[y], marker="o")
    for _, row in frame.iterrows():
        ax.annotate(str(int(row["epoch"])), (row[x], row[y]), fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or (f"{y} vs {x} (lr={lr})" if lr is not None else f"{y} vs {x}"))
    fig.tight_layout()
    return fig, ax
