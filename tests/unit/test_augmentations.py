"""Shape / dtype / seed reproducibility / NaN checks for every augmenter (spec 3.6)."""

import numpy as np
import pytest

from signal_aug.augmentations.methods import REGISTRY, apply_augmentation
from tests.fixtures.tiny import tiny_dataset

METHODS = sorted(REGISTRY)


@pytest.mark.parametrize("method", METHODS)
def test_output_shape_and_dtype(method):
    X, y = tiny_dataset()
    X_out, y_out = apply_augmentation(method, X, y, seed=0, params={})
    assert X_out.ndim == 3 and X_out.shape[1:] == X.shape[1:]
    assert X_out.dtype == np.float32
    assert y_out.dtype == np.int64
    assert len(X_out) == len(y_out)
    expected = len(X) if method == "none" else 2 * len(X)  # default ratio=1.0
    assert len(X_out) == expected


@pytest.mark.parametrize("method", METHODS)
def test_originals_preserved_and_labels_valid(method):
    X, y = tiny_dataset()
    X_out, y_out = apply_augmentation(method, X, y, seed=0, params={})
    np.testing.assert_array_equal(X_out[: len(X)], X)
    np.testing.assert_array_equal(y_out[: len(y)], y)
    assert set(np.unique(y_out)) <= set(np.unique(y))


@pytest.mark.parametrize("method", METHODS)
def test_no_nan_inf(method):
    X, y = tiny_dataset()
    X_out, _ = apply_augmentation(method, X, y, seed=0, params={})
    assert np.isfinite(X_out).all()


@pytest.mark.parametrize("method", METHODS)
def test_seed_reproducibility(method):
    X, y = tiny_dataset()
    X_a, y_a = apply_augmentation(method, X, y, seed=42, params={})
    X_b, y_b = apply_augmentation(method, X, y, seed=42, params={})
    np.testing.assert_array_equal(X_a, X_b)
    np.testing.assert_array_equal(y_a, y_b)


def test_different_seeds_differ():
    X, y = tiny_dataset()
    X_a, _ = apply_augmentation("jitter", X, y, seed=1, params={})
    X_b, _ = apply_augmentation("jitter", X, y, seed=2, params={})
    assert not np.array_equal(X_a, X_b)


def test_ratio_controls_synthetic_count():
    X, y = tiny_dataset()
    X_out, _ = apply_augmentation("jitter", X, y, seed=0, params={"ratio": 0.5})
    assert len(X_out) == len(X) + len(X) // 2


def test_unknown_method_raises():
    X, y = tiny_dataset()
    with pytest.raises(ValueError):
        apply_augmentation("nonexistent", X, y, seed=0)
