"""
Helper utilities for loading, saving, and standardizing RBP expression vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RBPExpression:
    """Container for an RBP expression vector."""

    values: np.ndarray
    names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        self.values = arr
        if self.names is not None:
            self.names = [str(name) for name in self.names]
            if len(self.names) != len(self.values):
                raise ValueError("Length mismatch between RBP names and values.")

    @property
    def dim(self) -> int:
        return int(self.values.shape[0])

    def to_dict(self) -> dict:
        data = {"values": self.values.tolist()}
        if self.names:
            data["rbp_names"] = self.names
        return data


def _load_json(path: Path) -> RBPExpression:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        values = data.get("values") or data.get("rbp_values")
        if values is None:
            raise ValueError(f"JSON file {path} must contain a 'values' array.")
        names = data.get("rbp_names")
    elif isinstance(data, Sequence):
        values = data
        names = None
    else:
        raise ValueError(f"Unsupported JSON structure in {path}.")
    return RBPExpression(values=np.asarray(values, dtype=np.float32), names=names)


def _load_npy(path: Path) -> RBPExpression:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "values" not in arr:
            raise ValueError(f"NPZ file {path} must have a 'values' array.")
        values = arr["values"]
        names = arr.get("rbp_names")
        if names is not None:
            names = [str(x) for x in names.tolist()]
    else:
        values = arr
        names = None
    return RBPExpression(values=values, names=names)


def load_rbp_expression(path: str | Path) -> RBPExpression:
    """
    Load an RBP expression vector from JSON/NPY/NPZ formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RBP expression file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix in {".npy", ".npz"}:
        return _load_npy(path)
    raise ValueError(f"Unsupported RBP expression format: {path.suffix}")


def save_rbp_expression(expression: RBPExpression, output: str | Path, fmt: str = "json") -> None:
    """
    Save an RBP expression vector to disk in JSON or NPY/NPZ format.
    """
    output_path = Path(output)
    fmt = fmt.lower()
    if fmt == "json":
        output_path.write_text(json.dumps(expression.to_dict(), indent=2))
    elif fmt == "npy":
        np.save(output_path, expression.values.astype(np.float32))
    elif fmt == "npz":
        np.savez(output_path, values=expression.values.astype(np.float32),
                 rbp_names=np.asarray(expression.names or [], dtype=object))
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def standardize_vector(values: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Apply optional normalization to RBP vectors.
    """
    arr = np.asarray(values, dtype=np.float32)
    if method == "none":
        return arr
    if method == "zscore":
        mean = arr.mean()
        std = arr.std()
        std = std if std > 0 else 1.0
        return (arr - mean) / std
    if method == "minmax":
        min_v = arr.min()
        max_v = arr.max()
        rng = max_v - min_v
        rng = rng if rng > 0 else 1.0
        return (arr - min_v) / rng
    raise ValueError(f"Unknown standardization method: {method}")
