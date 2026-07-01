from __future__ import annotations

import numpy as np


def representation_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute simple class-separation metrics in hidden representation space."""

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)
    classes = np.unique(y)
    if len(classes) < 2:
        return {"centroid_distance": np.nan, "within_class_distance": np.nan, "separation_ratio": np.nan}

    centroids = np.vstack([X[y == cls].mean(axis=0) for cls in classes[:2]])
    centroid_distance = float(np.linalg.norm(centroids[1] - centroids[0]))

    within = []
    for idx, cls in enumerate(classes[:2]):
        class_features = X[y == cls]
        if len(class_features):
            within.extend(np.linalg.norm(class_features - centroids[idx], axis=1))
    within_class_distance = float(np.mean(within)) if within else np.nan

    return {
        "centroid_distance": centroid_distance,
        "within_class_distance": within_class_distance,
        "separation_ratio": float(centroid_distance / (within_class_distance + 1e-12)),
    }
