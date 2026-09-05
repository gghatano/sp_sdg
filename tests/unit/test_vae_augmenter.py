"""Unit tests for the class-conditional VAE augmenter (issue #42, task GN-A1).

The properties that matter here are not "the samples look nice" but the ones
the experiment framework relies on: determinism given a seed, exact shape and
label bookkeeping, and -- the reason this augmenter exists at all -- that the
class condition actually steers generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_aug.augmentations.methods import REGISTRY, apply_augmentation


def _class_separated(n_per_class: int = 24, length: int = 32, seed: int = 0):
    """Three classes whose mean level is 0 / +4 / +8, so a generated sample's
    class is readable off its mean alone."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n_per_class)
    X = (rng.normal(scale=0.3, size=(len(y), 1, length)) + 4.0 * y[:, None, None]).astype(np.float32)
    return X, y


def test_vae_is_registered():
    assert "vae" in REGISTRY


def test_shapes_and_labels():
    X, y = _class_separated()
    X_aug, y_aug = apply_augmentation("vae", X, y, seed=1, params={"ratio": 1.0, "steps": 50})
    assert X_aug.shape == (2 * len(X), X.shape[1], X.shape[2])
    assert y_aug.shape == (2 * len(y),)
    assert X_aug.dtype == np.float32 and y_aug.dtype == np.int64
    # originals are preserved unchanged, up front
    assert np.array_equal(X_aug[: len(X)], X)
    assert np.array_equal(y_aug[: len(y)], y)
    assert np.isfinite(X_aug).all()


def test_same_seed_reproduces_identical_output():
    X, y = _class_separated()
    a, ya = apply_augmentation("vae", X, y, seed=7, params={"steps": 40})
    b, yb = apply_augmentation("vae", X, y, seed=7, params={"steps": 40})
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_different_seed_changes_output():
    X, y = _class_separated()
    a, _ = apply_augmentation("vae", X, y, seed=7, params={"steps": 40})
    c, _ = apply_augmentation("vae", X, y, seed=8, params={"steps": 40})
    assert not np.array_equal(a, c)


def test_ratio_controls_how_many_samples_are_added():
    X, y = _class_separated()
    for ratio, expected in ((0.0, 0), (0.5, len(X) // 2), (2.0, 2 * len(X))):
        X_aug, _ = apply_augmentation("vae", X, y, seed=1, params={"ratio": ratio, "steps": 30})
        assert len(X_aug) - len(X) == expected


def test_class_condition_steers_generation():
    """The point of a *conditional* VAE: a sample generated with label c must
    resemble class c. With classes separated by mean level, the generated
    sample's mean should sit nearest its own class's mean."""
    X, y = _class_separated(n_per_class=40, length=32)
    X_aug, y_aug = apply_augmentation("vae", X, y, seed=3, params={"ratio": 2.0, "steps": 600})
    synth_X, synth_y = X_aug[len(X):], y_aug[len(y):]
    class_means = np.array([X[y == c].mean() for c in (0, 1, 2)])

    nearest = np.abs(synth_X.mean(axis=(1, 2))[:, None] - class_means[None, :]).argmin(axis=1)
    hit_rate = (nearest == synth_y).mean()
    assert hit_rate > 0.8, f"conditioning did not steer generation (hit rate {hit_rate:.2f})"


def test_synthetic_class_mix_follows_training_prevalence():
    """Labels are drawn by picking source samples uniformly, so an imbalanced
    training set must yield a similarly imbalanced synthetic set (the
    oversample/smote convention, not the uniform-per-class one)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    X = rng.normal(size=(len(y), 1, 16)).astype(np.float32)
    _, y_aug = apply_augmentation("vae", X, y, seed=2, params={"ratio": 1.0, "steps": 30})
    synth = y_aug[len(y):]
    assert (synth == 0).mean() > 0.75, "synthetic labels should follow the 90/10 prevalence"


@pytest.mark.parametrize("length", [8, 24, 151])
def test_handles_short_odd_and_even_lengths(length):
    X, y = _class_separated(n_per_class=6, length=length)
    X_aug, _ = apply_augmentation("vae", X, y, seed=1, params={"steps": 20})
    assert X_aug.shape[2] == length and np.isfinite(X_aug).all()


def test_multichannel_input_is_preserved():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1] * 5)
    X = rng.normal(size=(len(y), 9, 24)).astype(np.float32)
    X_aug, _ = apply_augmentation("vae", X, y, seed=1, params={"steps": 20})
    assert X_aug.shape[1] == 9


def test_non_contiguous_labels_are_mapped_back():
    """Labels need not be 0..k-1; the augmenter maps to indices internally and
    must emit the ORIGINAL label values."""
    rng = np.random.default_rng(0)
    y = np.array([5, 5, 5, 9, 9, 9] * 4)
    X = rng.normal(size=(len(y), 1, 16)).astype(np.float32)
    _, y_aug = apply_augmentation("vae", X, y, seed=1, params={"steps": 20})
    assert set(np.unique(y_aug).tolist()) <= {5, 9}


def test_tiny_dataset_still_converges_on_a_step_budget():
    """The budget is optimizer STEPS, not epochs, precisely so a small dataset
    is not starved of training: 30 samples at batch 64 is one step per epoch,
    so an epoch budget would give it a fraction of the updates a large dataset
    gets. Same step budget -> conditioning works here too."""
    X, y = _class_separated(n_per_class=10, length=32)
    assert len(X) == 30
    X_aug, y_aug = apply_augmentation("vae", X, y, seed=3, params={"ratio": 2.0, "steps": 600})
    synth_X, synth_y = X_aug[len(X):], y_aug[len(y):]
    class_means = np.array([X[y == c].mean() for c in (0, 1, 2)])
    nearest = np.abs(synth_X.mean(axis=(1, 2))[:, None] - class_means[None, :]).argmin(axis=1)
    assert (nearest == synth_y).mean() > 0.8
