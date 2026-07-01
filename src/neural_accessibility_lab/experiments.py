from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_accessibility_lab.metrics import accessibility_metrics, hidden_hellinger_metrics, representation_metrics
from neural_accessibility_lab.models import SmallMLP
from neural_accessibility_lab.result import EpochState, ExperimentResult


def _positive_scores(logits) -> np.ndarray:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0]).detach().cpu().numpy()
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def run_mlp_accessibility_experiment(
    X,
    y,
    name: str = "mlp-accessibility",
    epochs: int = 50,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    dropout: float = 0.0,
    device=None,
) -> ExperimentResult:
    """Train ``SmallMLP`` and collect accessibility/representation metrics per epoch."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("run_mlp_accessibility_experiment requires torch. Install dependencies with `pip install -e .`.") from exc

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int64)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    torch.manual_seed(random_state)
    model = SmallMLP(X_train.shape[1], hidden_dim=hidden_dim, output_dim=2, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    x_val_tensor = torch.from_numpy(X_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).to(device)

    states: list[EpochState] = []
    checkpoints: dict = {}
    best_val_auc = -np.inf
    best_survival_auc = -np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train_tensor)
            val_logits, val_features = model(x_val_tensor, return_features=True)
            train_loss = float(criterion(train_logits, y_train_tensor).item())
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_scores = _positive_scores(val_logits)
            features_np = val_features.detach().cpu().numpy()

        val_auc = float(roc_auc_score(y_val, val_scores)) if len(np.unique(y_val)) == 2 else np.nan
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            **accessibility_metrics(y_val, val_scores),
            **representation_metrics(features_np, y_val),
            **hidden_hellinger_metrics(features_np, y_val),
        }

        states.append(
            EpochState(
                epoch=epoch,
                lr=float(optimizer.param_groups[0]["lr"]),
                metrics=metrics,
                features=features_np,
                labels=y_val.copy(),
            )
        )

        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoints["best_val_auc"] = deepcopy(model.state_dict())

        survival_auc = metrics.get("survival_auc", np.nan)
        if np.isfinite(survival_auc) and survival_auc > best_survival_auc:
            best_survival_auc = survival_auc
            checkpoints["best_survival_auc"] = deepcopy(model.state_dict())

    config = {
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "batch_size": batch_size,
        "test_size": test_size,
        "random_state": random_state,
        "dropout": dropout,
        "device": str(device),
        "scaler": scaler,
    }
    return ExperimentResult(name=name, config=config, states=states, checkpoints=checkpoints)
