"""TSTR (train on synthetic, test on real) -- issue #42, task GN-A2."""

from __future__ import annotations

import numpy as np
import pytest

from signal_aug.experiments.tstr import tstr_metrics
from signal_aug.reporting.aggregate import summarize

MODEL = ("cnn1d", {"epochs": 3, "hidden": 8})


def _separable(n_per_class: int = 24, length: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_per_class)
    X = (rng.normal(scale=0.2, size=(len(y), 1, length)) + 3.0 * y[:, None, None]).astype(np.float32)
    return X, y


def test_returns_prefixed_metrics():
    X, y = _separable()
    X_aug = np.concatenate([X, X])  # "synthetic" tail = copy of the originals
    y_aug = np.concatenate([y, y])
    out = tstr_metrics(X_aug, y_aug, len(y), X, y, *MODEL, seed=0)
    assert set(out) == {"tstr_accuracy", "tstr_macro_f1", "tstr_balanced_accuracy"}
    assert all(0.0 <= v <= 1.0 for v in out.values())
    # n_test belongs to the real evaluation, not to the TSTR block
    assert "tstr_n_test" not in out


def test_no_synthetic_samples_yields_nothing():
    """`none` adds no synthetic tail, so there is nothing to train on."""
    X, y = _separable()
    assert tstr_metrics(X, y, len(y), X, y, *MODEL, seed=0) == {}


def test_single_class_synthetic_set_yields_nothing():
    X, y = _separable()
    X_aug = np.concatenate([X, X[y == 0]])
    y_aug = np.concatenate([y, y[y == 0]])
    assert tstr_metrics(X_aug, y_aug, len(y), X, y, *MODEL, seed=0) == {}


def test_only_the_synthetic_tail_is_used_for_fitting():
    """A synthetic tail carrying the labels INVERTED must score near zero:
    proof the real head was not used to fit."""
    X, y = _separable(n_per_class=40)
    X_aug = np.concatenate([X, X])
    y_aug = np.concatenate([y, 1 - y])  # tail: deliberately wrong labels
    out = tstr_metrics(X_aug, y_aug, len(y), X, y, *MODEL, seed=0)
    assert out["tstr_accuracy"] < 0.5, "the real (correctly labelled) head leaked into the fit"


def test_summarize_includes_tstr_when_every_seed_has_it():
    rows = [
        {"status": "completed", "dataset": "D", "augmentation": "vae", "model": "m", "train_fraction": 1.0,
         "accuracy": 0.8, "macro_f1": 0.8, "balanced_accuracy": 0.8,
         "tstr_accuracy": 0.6, "tstr_macro_f1": 0.6, "tstr_balanced_accuracy": 0.6},
        {"status": "completed", "dataset": "D", "augmentation": "vae", "model": "m", "train_fraction": 1.0,
         "accuracy": 0.9, "macro_f1": 0.9, "balanced_accuracy": 0.9,
         "tstr_accuracy": 0.7, "tstr_macro_f1": 0.7, "tstr_balanced_accuracy": 0.7},
    ]
    entry = summarize(rows)[0]
    assert entry["tstr_accuracy_mean"] == pytest.approx(0.65)
    assert entry["accuracy_mean"] == pytest.approx(0.85)


def test_summarize_omits_tstr_when_only_some_seeds_have_it():
    """Averaging a half-populated cell would silently use a different seed set
    than the headline metric, so it is dropped instead."""
    rows = [
        {"status": "completed", "dataset": "D", "augmentation": "vae", "model": "m", "train_fraction": 1.0,
         "accuracy": 0.8, "macro_f1": 0.8, "balanced_accuracy": 0.8, "tstr_accuracy": 0.6,
         "tstr_macro_f1": 0.6, "tstr_balanced_accuracy": 0.6},
        {"status": "completed", "dataset": "D", "augmentation": "vae", "model": "m", "train_fraction": 1.0,
         "accuracy": 0.9, "macro_f1": 0.9, "balanced_accuracy": 0.9},
    ]
    entry = summarize(rows)[0]
    assert "tstr_accuracy_mean" not in entry
    assert entry["accuracy_mean"] == pytest.approx(0.85)


def test_grid_carries_the_config_flag():
    from signal_aug.experiments.runner import expand_grid

    exp = {"name": "x", "phase": 9, "datasets": ["D"], "augmentations": ["a"],
           "models": ["m"], "seeds": [0]}
    aug = {"augmentations": {"a": {"method": "jitter", "params": {}}}}
    mdl = {"models": {"m": {"type": "cnn1d", "params": {}}}}
    assert expand_grid(exp, aug, mdl)[0].tstr is False
    assert expand_grid({**exp, "tstr": True}, aug, mdl)[0].tstr is True
