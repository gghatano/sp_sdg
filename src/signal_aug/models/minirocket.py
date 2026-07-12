"""MiniRocket features + RidgeClassifierCV (the standard MiniRocket pipeline).

RidgeClassifierCV uses internal (training-data-only) cross-validation, so no
test data touches hyperparameter selection.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV


class MiniRocketClassifier:
    def __init__(self, n_kernels: int = 10000, seed: int = 0):
        self.n_kernels = n_kernels
        self.seed = seed
        self.transformer = None
        self.classifier = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketClassifier":
        from aeon.transformations.collection.convolution_based import MiniRocket

        self.transformer = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed)
        features = self.transformer.fit_transform(X.astype(np.float64))
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.transformer is None or self.classifier is None:
            raise RuntimeError("fit() must be called before predict()")
        features = self.transformer.transform(X.astype(np.float64))
        return self.classifier.predict(features).astype(np.int64)
