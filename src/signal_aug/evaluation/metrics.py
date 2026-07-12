"""Evaluation metrics for classification runs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

METRIC_NAMES = ["accuracy", "macro_f1", "balanced_accuracy"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }
