from __future__ import annotations

import pandas as pd


def compute_lag_correlations(frame, predictor_cols, target_col: str = "delta_survival_auc", max_lag: int = 5) -> pd.DataFrame:
    """Compute correlations between lagged predictors and a target metric."""

    rows = []
    for predictor in predictor_cols:
        for lag in range(max_lag + 1):
            shifted = frame[predictor].shift(lag)
            valid = shifted.notna() & frame[target_col].notna()
            x = shifted[valid]
            y = frame.loc[valid, target_col]
            if valid.sum() >= 2 and x.nunique() > 1 and y.nunique() > 1:
                corr = x.corr(y)
            else:
                corr = float("nan")
            rows.append({"predictor": predictor, "target": target_col, "lag": lag, "correlation": corr})
    return pd.DataFrame(rows)
