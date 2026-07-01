from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _binary_scores(y_score: np.ndarray) -> np.ndarray:
    scores = np.asarray(y_score)
    if scores.ndim == 2:
        if scores.shape[1] == 1:
            return scores[:, 0]
        return scores[:, 1]
    return scores


def accessibility_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | None = None) -> dict[str, float]:
    """Compute threshold-survival style accessibility metrics for binary labels."""

    y = np.asarray(y_true).astype(int)
    scores = _binary_scores(y_score)
    thresholds = np.asarray(thresholds) if thresholds is not None else np.linspace(0.0, 1.0, 101)

    positives = y == 1
    negatives = ~positives
    auc = np.nan if positives.sum() == 0 or negatives.sum() == 0 else float(roc_auc_score(y, scores))

    survival = np.array([(scores[positives] >= t).mean() if positives.any() else np.nan for t in thresholds])
    false_access = np.array([(scores[negatives] >= t).mean() if negatives.any() else np.nan for t in thresholds])

    return {
        "auc": auc,
        "survival_auc": float(np.trapz(survival, thresholds)),
        "mean_positive_score": float(scores[positives].mean()) if positives.any() else np.nan,
        "mean_negative_score": float(scores[negatives].mean()) if negatives.any() else np.nan,
        "false_access_auc": float(np.trapz(false_access, thresholds)),
    }
