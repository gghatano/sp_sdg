"""Chance floor reported with every run (issue #48).

Accuracy on an imbalanced test set is unreadable without the majority rate:
Wafer's test split is 89.2% one class, so 0.93 accuracy is nearly the
always-predict-majority floor even though it looks high.
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_aug.evaluation.metrics import compute_metrics
from signal_aug.experiments.tstr import tstr_metrics
from signal_aug.reporting.aggregate import summarize


def test_majority_rate_matches_the_test_distribution():
    y = np.array([1] * 89 + [0] * 11)
    m = compute_metrics(y, np.ones(100, dtype=int))
    assert m["majority_rate"] == pytest.approx(0.89)
    # the degenerate classifier scores at exactly the floor
    assert m["accuracy"] == pytest.approx(m["majority_rate"])
    # ...and balanced accuracy exposes it
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_balanced_test_set_floor_is_one_over_n_classes():
    y = np.array([0] * 50 + [1] * 50)
    assert compute_metrics(y, np.zeros(100, dtype=int))["majority_rate"] == pytest.approx(0.5)


def test_three_class_floor_is_not_one_third():
    y = np.array([0] * 70 + [1] * 20 + [2] * 10)
    assert compute_metrics(y, np.zeros(100, dtype=int))["majority_rate"] == pytest.approx(0.7)


def test_tstr_does_not_duplicate_test_set_descriptors():
    """n_test and majority_rate describe the shared test set, so the TSTR block
    must not restate them with a tstr_ prefix."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    X = (rng.normal(scale=0.2, size=(40, 1, 16)) + 3.0 * y[:, None, None]).astype(np.float32)
    out = tstr_metrics(np.concatenate([X, X]), np.concatenate([y, y]), len(y),
                       X, y, "cnn1d", {"epochs": 3, "hidden": 8}, seed=0)
    assert "tstr_n_test" not in out and "tstr_majority_rate" not in out
    assert set(out) == {"tstr_accuracy", "tstr_macro_f1", "tstr_balanced_accuracy"}


def test_summarize_carries_the_floor():
    rows = [
        {"status": "completed", "dataset": "Wafer", "augmentation": "vae", "model": "m",
         "train_fraction": 1.0, "accuracy": 0.93, "macro_f1": 0.7, "balanced_accuracy": 0.70,
         "majority_rate": 0.892},
        {"status": "completed", "dataset": "Wafer", "augmentation": "vae", "model": "m",
         "train_fraction": 1.0, "accuracy": 0.94, "macro_f1": 0.7, "balanced_accuracy": 0.69,
         "majority_rate": 0.892},
    ]
    entry = summarize(rows)[0]
    assert entry["majority_rate"] == pytest.approx(0.892)
    # the point of the issue: accuracy sits just above the floor while balanced
    # accuracy shows the minority class is barely learned
    assert entry["accuracy_mean"] - entry["majority_rate"] < 0.05
    assert entry["balanced_accuracy_mean"] < 0.75


def test_summarize_omits_floor_when_older_runs_lack_it():
    rows = [
        {"status": "completed", "dataset": "D", "augmentation": "none", "model": "m",
         "train_fraction": 1.0, "accuracy": 0.9, "macro_f1": 0.9, "balanced_accuracy": 0.9,
         "majority_rate": 0.5},
        {"status": "completed", "dataset": "D", "augmentation": "none", "model": "m",
         "train_fraction": 1.0, "accuracy": 0.9, "macro_f1": 0.9, "balanced_accuracy": 0.9},
    ]
    assert "majority_rate" not in summarize(rows)[0]
