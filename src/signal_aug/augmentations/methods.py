"""Training-set augmentation methods.

Every augmenter has the signature
    augment(X, y, rng, **params) -> (X_out, y_out)
where X is (n, channels, length) float32 and y is (n,) int64. The output
contains the original samples followed by synthetic ones. Augmenters only
ever receive training data; passing test data is a spec violation (section 8).

`ratio` is the number of synthetic samples added as a fraction of n.
Synthetic samples always inherit the label of their source sample(s), so no
soft labels are produced (mixup is restricted to same-class pairs; recorded
in artifacts/deviations.md).
"""

from __future__ import annotations

import numpy as np


def _n_synthetic(n: int, ratio: float) -> int:
    return int(round(n * ratio))


def _stack(X: np.ndarray, y: np.ndarray, X_new: list[np.ndarray], y_new: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not X_new:
        return X.astype(np.float32), y.astype(np.int64)
    X_out = np.concatenate([X, np.stack(X_new).astype(np.float32)], axis=0)
    y_out = np.concatenate([y, np.asarray(y_new, dtype=np.int64)], axis=0)
    return X_out, y_out


def augment_none(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return X.astype(np.float32), y.astype(np.int64)


def augment_oversample(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(X), size=_n_synthetic(len(X), ratio))
    return _stack(X, y, [X[i].copy() for i in idx], [int(y[i]) for i in idx])


def augment_jitter(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0, sigma: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(X), size=_n_synthetic(len(X), ratio))
    X_new = [X[i] + rng.normal(0, sigma, size=X[i].shape).astype(np.float32) for i in idx]
    return _stack(X, y, X_new, [int(y[i]) for i in idx])


def augment_scaling(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0, sigma: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.integers(0, len(X), size=_n_synthetic(len(X), ratio))
    X_new = []
    for i in idx:
        factor = rng.normal(1.0, sigma, size=(X.shape[1], 1)).astype(np.float32)
        X_new.append(X[i] * factor)
    return _stack(X, y, X_new, [int(y[i]) for i in idx])


def _same_class_pairs(y: np.ndarray, rng: np.random.Generator, n_pairs: int) -> list[tuple[int, int]]:
    by_class = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}
    eligible = [c for c, idx in by_class.items() if len(idx) >= 2]
    pairs = []
    for _ in range(n_pairs):
        if eligible:
            cls = int(rng.choice(eligible))
            i, j = rng.choice(by_class[cls], size=2, replace=False)
        else:
            # degenerate case: classes with a single sample; pair with itself
            cls = int(rng.choice(list(by_class)))
            i = j = by_class[cls][0]
        pairs.append((int(i), int(j)))
    return pairs


def augment_mixup(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0, alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Same-class mixup: convex combination of two samples from one class."""
    pairs = _same_class_pairs(y, rng, _n_synthetic(len(X), ratio))
    X_new, y_new = [], []
    for i, j in pairs:
        lam = float(rng.beta(alpha, alpha))
        X_new.append((lam * X[i] + (1.0 - lam) * X[j]).astype(np.float32))
        y_new.append(int(y[i]))
    return _stack(X, y, X_new, y_new)


def augment_dtw(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0, window: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """DTW-aligned averaging of same-class pairs.

    The second series is warped onto the first via the DTW alignment path,
    then the two are averaged. Produces a plausible intermediate sample.
    """
    from aeon.distances import dtw_alignment_path

    pairs = _same_class_pairs(y, rng, _n_synthetic(len(X), ratio))
    length = X.shape[2]
    X_new, y_new = [], []
    for i, j in pairs:
        sample = np.empty_like(X[i])
        for c in range(X.shape[1]):
            a, b = X[i, c].astype(np.float64), X[j, c].astype(np.float64)
            path, _ = dtw_alignment_path(a, b, window=window)
            aligned = np.zeros(length)
            counts = np.zeros(length)
            for pa, pb in path:
                aligned[pa] += b[pb]
                counts[pa] += 1
            aligned /= np.maximum(counts, 1)
            sample[c] = ((a + aligned) / 2.0).astype(np.float32)
        X_new.append(sample)
        y_new.append(int(y[i]))
    return _stack(X, y, X_new, y_new)


def augment_smote(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Raw-SMOTE: interpolate towards a same-class k-nearest neighbour in raw signal space."""
    from sklearn.neighbors import NearestNeighbors

    n_new = _n_synthetic(len(X), ratio)
    flat = X.reshape(len(X), -1)
    by_class = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}
    knn_by_class = {}
    for cls, idx in by_class.items():
        n_neighbors = min(k + 1, len(idx))
        knn_by_class[cls] = (idx, NearestNeighbors(n_neighbors=n_neighbors).fit(flat[idx]))

    X_new, y_new = [], []
    for _ in range(n_new):
        i = int(rng.integers(0, len(X)))
        cls = int(y[i])
        idx, knn = knn_by_class[cls]
        _, neigh = knn.kneighbors(flat[i : i + 1])
        candidates = [idx[j] for j in neigh[0] if idx[j] != i]
        j = int(rng.choice(candidates)) if candidates else i
        lam = float(rng.uniform(0, 1))
        X_new.append((X[i] + lam * (X[j] - X[i])).astype(np.float32))
        y_new.append(cls)
    return _stack(X, y, X_new, y_new)


def augment_label_shuffle(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Negative control (spec section 5, Phase 5): copy real samples but assign
    RANDOM labels. This injects label noise and carries no genuine class signal,
    so a sound pipeline must NOT show a subject-count reduction for it. If it did,
    the measured reductions would be suspect artifacts."""
    idx = rng.integers(0, len(X), size=_n_synthetic(len(X), ratio))
    classes = np.unique(y)
    X_new = [X[i].copy() for i in idx]
    y_new = [int(rng.choice(classes)) for _ in idx]
    return _stack(X, y, X_new, y_new)


REGISTRY = {
    "none": augment_none,
    "oversample": augment_oversample,
    "jitter": augment_jitter,
    "scaling": augment_scaling,
    "mixup": augment_mixup,
    "dtw": augment_dtw,
    "smote": augment_smote,
    "label_shuffle": augment_label_shuffle,
}


def apply_augmentation(
    method: str, X: np.ndarray, y: np.ndarray, seed: int, params: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if method not in REGISTRY:
        raise ValueError(f"Unknown augmentation method: {method}")
    rng = np.random.default_rng(seed)
    return REGISTRY[method](X, y, rng, **(params or {}))
