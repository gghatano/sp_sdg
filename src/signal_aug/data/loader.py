"""Dataset loading: UCR archive (via aeon, cached in data/raw) and synthetic data.

All arrays are float32 with shape (n_cases, n_channels, n_timepoints).
Labels are int64 class indices. Test data is returned separately and must
never be passed to augmenters or fitting code (spec section 8).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetSplits:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]

    @property
    def dataset_checksum(self) -> str:
        h = hashlib.sha256()
        for arr in (self.X_train, self.y_train, self.X_test, self.y_test):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    @property
    def split_checksum(self) -> str:
        h = hashlib.sha256()
        h.update(f"{len(self.y_train)}:{len(self.y_test)}".encode())
        h.update(np.ascontiguousarray(self.y_train).tobytes())
        h.update(np.ascontiguousarray(self.y_test).tobytes())
        return h.hexdigest()


def _znorm(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def _encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted(set(np.asarray(y_train).tolist()) | set(np.asarray(y_test).tolist()))
    mapping = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([mapping[v] for v in np.asarray(y).tolist()], dtype=np.int64)
    return enc(y_train), enc(y_test), [str(c) for c in classes]


def load_synthetic(spec: dict, seed: int = 12345) -> DatasetSplits:
    """Two-class synthetic dataset: noisy sine vs. noisy square wave.

    Deterministic for a given spec+seed so checksums are stable across runs.
    """
    rng = np.random.default_rng(seed)
    n_train, n_test = int(spec["n_train"]), int(spec["n_test"])
    length = int(spec["length"])
    n_channels = int(spec.get("n_channels", 1))

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, 2, size=n).astype(np.int64)
        t = np.linspace(0, 4 * np.pi, length, dtype=np.float32)
        X = np.zeros((n, n_channels, length), dtype=np.float32)
        for i in range(n):
            phase = rng.uniform(0, 2 * np.pi)
            base = np.sin(t + phase) if y[i] == 0 else np.sign(np.sin(t + phase))
            for c in range(n_channels):
                X[i, c] = base + rng.normal(0, 0.3, size=length)
        return X.astype(np.float32), y

    X_train, y_train = make(n_train)
    X_test, y_test = make(n_test)
    return DatasetSplits("synthetic", _znorm(X_train), y_train, _znorm(X_test), y_test, ["sine", "square"])


def load_ucr(name: str, data_dir: str | Path = "data/raw") -> DatasetSplits:
    from aeon.datasets import load_classification

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_classification(name, split="train", extract_path=str(data_dir))
    X_test, y_test = load_classification(name, split="test", extract_path=str(data_dir))
    y_train, y_test, class_names = _encode_labels(y_train, y_test)
    return DatasetSplits(
        name,
        _znorm(X_train.astype(np.float32)),
        y_train,
        _znorm(X_test.astype(np.float32)),
        y_test,
        class_names,
    )


def load_dataset(name: str, datasets_config: dict, data_dir: str | Path = "data/raw") -> DatasetSplits:
    spec = datasets_config["datasets"][name]
    if spec["source"] == "synthetic":
        return load_synthetic(spec)
    if spec["source"] == "ucr":
        return load_ucr(name, data_dir=data_dir)
    raise ValueError(f"Unknown dataset source: {spec['source']}")


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation split. Operates on training data only."""
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]
