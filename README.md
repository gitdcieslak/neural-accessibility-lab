# Neural Accessibility Lab

Reusable, notebook-friendly experiment harness for studying neural-network accessibility dynamics.

## Quickstart

```bash
pip install -e .
```

```python
from neural_accessibility_lab.datasets import load_oil
from neural_accessibility_lab.experiments import run_mlp_accessibility_experiment

X, y = load_oil()

result = run_mlp_accessibility_experiment(
    X,
    y,
    name="oil-small-mlp",
    epochs=25,
)

frame = result.to_frame()
result.plot_drift_dashboard()
```

The package is intentionally small and keeps notebook workflows intact. Use the modules to clean up notebooks without replacing exploratory analysis.

## Main imports

```python
from neural_accessibility_lab.datasets import load_oil
from neural_accessibility_lab.models import SmallMLP
from neural_accessibility_lab.experiments import run_mlp_accessibility_experiment
from neural_accessibility_lab.metrics import accessibility_metrics, hellinger_1d, centroid_drift
from neural_accessibility_lab.analyses import compute_lag_correlations
```

The old short package name is kept as a lightweight compatibility alias for now, so existing notebooks using `import nal` can be migrated gradually.

## Data

`data/oil.data` is mirrored into the package as `src/neural_accessibility_lab/data/oil.data` so `load_oil()` works after installation. Replace both files with the full dataset if needed.
