"""Model factory. Adding a model is one entry in REGISTRY (mirrors the
augmentation REGISTRY). Classifiers implement fit(X, y) -> self and
predict(X) -> np.ndarray."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _build_cnn1d(params: dict, seed: int) -> Classifier:
    from signal_aug.models.cnn1d import CNN1DClassifier

    return CNN1DClassifier(seed=seed, **params)


def _build_minirocket(params: dict, seed: int) -> Classifier:
    from signal_aug.models.minirocket import MiniRocketClassifier

    return MiniRocketClassifier(seed=seed, **params)


REGISTRY = {
    "cnn1d": _build_cnn1d,
    "minirocket": _build_minirocket,
}


def build_model(model_type: str, params: dict, seed: int) -> Classifier:
    if model_type not in REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return REGISTRY[model_type](params, seed)
