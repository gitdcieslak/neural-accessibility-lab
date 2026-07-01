"""Neural Accessibility Lab."""

__version__ = "0.1.1"

__all__ = ["EpochState", "ExperimentResult", "SmallMLP", "__version__", "load_oil", "run_mlp_accessibility_experiment"]


def __getattr__(name: str):
    if name == "load_oil":
        from neural_accessibility_lab.datasets import load_oil

        return load_oil
    if name == "SmallMLP":
        from neural_accessibility_lab.models import SmallMLP

        return SmallMLP
    if name in {"EpochState", "ExperimentResult"}:
        from neural_accessibility_lab import result

        return getattr(result, name)
    if name == "run_mlp_accessibility_experiment":
        from neural_accessibility_lab.experiments import run_mlp_accessibility_experiment

        return run_mlp_accessibility_experiment
    raise AttributeError(f"module 'neural_accessibility_lab' has no attribute {name!r}")
