from neural_accessibility_lab.datasets import load_oil
from neural_accessibility_lab.experiments import run_mlp_accessibility_experiment
from neural_accessibility_lab.models import SmallMLP

DATASETS = {"oil": load_oil}
MODELS = {"small_mlp": SmallMLP}
EXPERIMENTS = {"mlp_accessibility": run_mlp_accessibility_experiment}

__all__ = ["DATASETS", "EXPERIMENTS", "MODELS"]
