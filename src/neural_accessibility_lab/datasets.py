from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


def _oil_path() -> Path:
    packaged = resources.files("neural_accessibility_lab").joinpath("data/oil.data")
    if packaged.is_file():
        return Path(str(packaged))

    root_data = Path(__file__).resolve().parents[2] / "data" / "oil.data"
    if root_data.exists():
        return root_data

    raise FileNotFoundError("Could not find oil.data in package data or repository data directory.")


def load_oil(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the oil dataset as ``(X, y)``.

    The loader assumes the target is the last column and accepts comma or whitespace
    separated files. Comment lines starting with ``#`` are ignored.
    """

    data_path = Path(path) if path is not None else _oil_path()
    df = pd.read_csv(data_path, sep=r"[\s,]+", comment="#", header=None, engine="python")
    if df.empty or df.shape[1] < 2:
        raise ValueError(f"Expected at least one feature column and one target column in {data_path}.")

    values = df.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"Found non-numeric values in {data_path}.")

    X = values.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = values.iloc[:, -1].to_numpy(dtype=np.int64)
    return X, y
