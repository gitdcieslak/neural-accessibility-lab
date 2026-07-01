import importlib.util
import unittest

from neural_accessibility_lab.datasets import load_oil
from neural_accessibility_lab.experiments import run_mlp_accessibility_experiment


class ExperimentSmokeTest(unittest.TestCase):
    def test_experiments_import_without_running_torch(self):
        import neural_accessibility_lab.experiments as experiments

        self.assertTrue(hasattr(experiments, "run_mlp_accessibility_experiment"))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "full experiment smoke test requires torch")
    def test_run_mlp_accessibility_experiment_smoke(self):
        X, y = load_oil()
        result = run_mlp_accessibility_experiment(X, y, lr=0.001, epochs=2)
        frame = result.to_frame()
        self.assertEqual(len(frame), 2)


if __name__ == "__main__":
    unittest.main()
