"""Metric range and mismatch checks (spec 3.6)."""

import numpy as np
import pytest

from signal_aug.evaluation.metrics import compute_metrics


def test_perfect_prediction():
    y = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["n_test"] == 4


def test_metrics_in_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    y_pred = rng.integers(0, 3, 100)
    metrics = compute_metrics(y_true, y_pred)
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert 0.0 <= metrics[name] <= 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0]))


def test_metrics_degenerate_single_class_prediction_no_nan():
    import numpy as np
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.zeros(5, dtype=np.int64)  # predicts only class 0
    m = compute_metrics(y_true, y_pred)
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        assert 0.0 <= m[name] <= 1.0
        assert m[name] == m[name]  # not NaN


def test_metrics_all_correct_multiclass():
    import numpy as np
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["balanced_accuracy"] == 1.0
