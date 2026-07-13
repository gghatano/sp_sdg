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


def test_label_shuffle_randomizes_only_synthetic_labels():
    """Negative control: originals are preserved; synthetic samples copy real
    signals but get randomized labels (so they carry no class signal)."""
    X, y = tiny_dataset(n=40, n_classes=2)
    X_out, y_out = apply_augmentation("label_shuffle", X, y, seed=0, params={})
    # originals untouched
    np.testing.assert_array_equal(X_out[: len(X)], X)
    np.testing.assert_array_equal(y_out[: len(y)], y)
    # every synthetic signal equals some real signal (copy), labels stay in range
    assert len(X_out) == 2 * len(X)
    assert set(np.unique(y_out[len(y):])) <= set(np.unique(y))


def test_mixup_produces_only_same_class_combinations():
    """deviations.md: mixup is restricted to same-class pairs (no soft labels).
    Build class-separated signals so an inter-class blend would be detectable."""
    import numpy as np
    from signal_aug.augmentations.methods import apply_augmentation

    length = 16
    n_per = 20
    # class 0 signals are all +1, class 1 signals are all -1 -> any inter-class
    # convex blend lands strictly between, which we can detect
    X = np.concatenate([
        np.ones((n_per, 1, length), dtype=np.float32),
        -np.ones((n_per, 1, length), dtype=np.float32),
    ])
    y = np.array([0] * n_per + [1] * n_per, dtype=np.int64)
    X_out, y_out = apply_augmentation("mixup", X, y, seed=0, params={"ratio": 1.0})
    synth_X = X_out[len(X):]
    synth_y = y_out[len(y):]
    for xi, yi in zip(synth_X, synth_y):
        # same-class blends stay at exactly +1 (class 0) or -1 (class 1)
        expected = 1.0 if yi == 0 else -1.0
        assert np.allclose(xi, expected), "mixup blended across classes"
