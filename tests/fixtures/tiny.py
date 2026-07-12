"""Tiny deterministic fixture data for unit tests (no network, no disk)."""

import numpy as np


def tiny_dataset(n: int = 24, channels: int = 1, length: int = 32, n_classes: int = 2, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, channels, length)).astype(np.float32)
    y = np.arange(n, dtype=np.int64) % n_classes
    return X, y
