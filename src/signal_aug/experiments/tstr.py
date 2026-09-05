"""TSTR: Train on Synthetic, Test on Real (issue #42, task GN-A2).

The headline metric of an augmentation experiment says whether adding synthetic
samples to the real ones helped. It cannot say WHY a method failed: a generator
that produces garbage and a generator that produces good-but-redundant samples
both show up as "no improvement".

TSTR separates the two. A classifier is trained on the synthetic samples ALONE
and evaluated on the real test set. Near-chance TSTR means the generator failed;
healthy TSTR alongside a flat headline metric means the generator works but adds
nothing the real samples did not already carry.

This trains a second classifier per run, so it is opt-in per experiment config
(`tstr: true`). No test data reaches any fit: the synthetic samples come from an
augmenter that only ever saw the training split.
"""

from __future__ import annotations

import numpy as np

from signal_aug.evaluation.metrics import compute_metrics
from signal_aug.models import build_model


def tstr_metrics(
    X_aug: np.ndarray,
    y_aug: np.ndarray,
    n_real: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    model_params: dict,
    seed: int,
) -> dict:
    """Fit a fresh classifier on the synthetic tail of (X_aug, y_aug) only.

    Returns metrics prefixed with `tstr_`, or {} when the synthetic set cannot
    support a classifier (no synthetic samples, or only one class in them).
    Augmenters append their synthetic samples after the originals, so the tail
    from n_real onwards is exactly the synthetic set.
    """
    X_syn, y_syn = X_aug[n_real:], y_aug[n_real:]
    if len(y_syn) < 2 or len(np.unique(y_syn)) < 2:
        return {}

    model = build_model(model_type, model_params, seed=seed)
    model.fit(X_syn, y_syn)
    scores = compute_metrics(y_test, model.predict(X_test))
    return {f"tstr_{k}": v for k, v in scores.items() if k != "n_test"}
