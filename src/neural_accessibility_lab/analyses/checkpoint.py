from __future__ import annotations

import pandas as pd


def checkpoint_summary(result) -> pd.DataFrame:
    """Summarize available checkpoints for a result."""

    frame = result.to_frame()
    rows = []
    for name in result.checkpoints:
        metric = name.replace("best_", "")
        row = {"checkpoint": name, "metric": metric}
        if metric in frame.columns and not frame.empty:
            idx = frame[metric].idxmax()
            row.update({"epoch": int(frame.loc[idx, "epoch"]), "value": float(frame.loc[idx, metric])})
        rows.append(row)
    return pd.DataFrame(rows)
