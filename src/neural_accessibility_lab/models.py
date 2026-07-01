from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional runtime deps
    torch = None
    nn = None


if nn is None:

    class SmallMLP:  # type: ignore[no-redef]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("SmallMLP requires torch. Install package dependencies with `pip install -e .`.")

else:

    class SmallMLP(nn.Module):
        """Small fully connected classifier with optional hidden features."""

        def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 2, dropout: float = 0.0):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor, return_features: bool = False):
            features = self.features(x)
            logits = self.classifier(features)
            if return_features:
                return logits, features
            return logits
