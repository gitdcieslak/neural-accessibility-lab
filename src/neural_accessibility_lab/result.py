from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EpochState:
    epoch: int
    lr: float
    metrics: dict
    features: np.ndarray
    labels: np.ndarray


@dataclass
class ExperimentResult:
    name: str
    config: dict
    states: list[EpochState]
    checkpoints: dict

    def to_frame(self) -> pd.DataFrame:
        rows = [{"epoch": state.epoch, "lr": state.lr, **state.metrics} for state in self.states]
        return pd.DataFrame(rows)

    def build_drift_dashboard_frame(self) -> pd.DataFrame:
        from neural_accessibility_lab.analyses.dashboard import build_drift_dashboard_frame

        return build_drift_dashboard_frame(self)

    def plot_drift_dashboard(self, lr=None):
        from neural_accessibility_lab.analyses.dashboard import plot_drift_dashboard

        return plot_drift_dashboard(self, lr=lr)

    def compute_lag_correlations(self, predictor_cols, target_col: str = "delta_survival_auc", max_lag: int = 5):
        from neural_accessibility_lab.analyses.lag import compute_lag_correlations

        return compute_lag_correlations(self.build_drift_dashboard_frame(), predictor_cols, target_col, max_lag)

    def plot_phase_portrait(self, x, y, lr=None, title=None):
        from neural_accessibility_lab.analyses.phase_portraits import plot_phase_portrait

        return plot_phase_portrait(self.build_drift_dashboard_frame(), x, y, lr=lr, title=title)

    def checkpoint_summary(self):
        from neural_accessibility_lab.analyses.checkpoint import checkpoint_summary

        return checkpoint_summary(self)
