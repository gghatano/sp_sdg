"""Negative-control leakage tests: these must FAIL if test data reaches fit,
or if a test subject reaches the training pool (spec section 8, CLAUDE.md).

Unlike the data-layer duplicate checks, these exercise the runner path and are
designed to break if leakage were introduced.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from signal_aug.data.loader import DatasetSplits
from signal_aug.data.subject import SubjectSplits
from signal_aug.experiments.runner import RunSpec, execute_run
from signal_aug.experiments.subject_runner import SubjectRunSpec, execute_subject_run
from signal_aug.models import factory


class _SpyModel:
    """Records every X passed to fit(), so a test can assert test data never
    reached fitting."""

    fit_inputs: list = []

    def __init__(self, seed=0):
        self.classes = None

    def fit(self, X, y):
        _SpyModel.fit_inputs.append(np.array(X, copy=True))
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


@pytest.fixture
def spy_model(monkeypatch):
    _SpyModel.fit_inputs = []
    monkeypatch.setitem(factory.REGISTRY, "spy", lambda params, seed: _SpyModel(seed=seed))
    return _SpyModel


def _rows_to_set(X):
    return {row.tobytes() for row in np.ascontiguousarray(X).reshape(len(X), -1)}


def test_execute_run_never_fits_on_test_data(spy_model, synthetic_dataset, tmp_path):
    """The runner must not pass any test sample to model.fit() (§8 item: fit)."""
    spec = RunSpec(
        run_id="leak_grid", phase=0, dataset="synthetic",
        augmentation="jitter", augmentation_params={"ratio": 1.0},
        model="spy", model_type="spy", model_params={}, seed=0,
    )
    manifest = execute_run(spec, synthetic_dataset, runs_dir=tmp_path)
    assert manifest["status"] == "completed"
    assert spy_model.fit_inputs, "fit was never called"

    fit_rows = set()
    for X in spy_model.fit_inputs:
        fit_rows |= _rows_to_set(X)
    test_rows = _rows_to_set(synthetic_dataset.X_test)
    assert not (fit_rows & test_rows), "a test sample reached model.fit() — leakage"


def test_execute_run_fit_size_matches_augmented_train_only(spy_model, synthetic_dataset, tmp_path):
    """Sanity: fit sees train (+synthetic), never the extra test rows."""
    spec = RunSpec(
        run_id="leak_grid2", phase=0, dataset="synthetic",
        augmentation="none", augmentation_params={},
        model="spy", model_type="spy", model_params={}, seed=0,
    )
    execute_run(spec, synthetic_dataset, runs_dir=tmp_path)
    fit_n = len(spy_model.fit_inputs[-1])
    assert fit_n == len(synthetic_dataset.X_train)  # none augmentation, train only


def _leaky_pool_and_test():
    """Pool and test that deliberately share ALL subjects, so whichever subject
    the runner selects for training is guaranteed to overlap the test set."""
    rng = np.random.default_rng(1)
    n, length = 30, 16
    X = rng.normal(0, 1, size=(n, 1, length)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    subjects = np.array([i % 3 for i in range(n)])  # subjects 0,1,2
    pool = SubjectSplits("pool", X, y, subjects)
    test = SubjectSplits("test", X[:12], y[:12], np.array([i % 3 for i in range(12)]))
    return pool, test


def test_subject_run_guard_rejects_leaked_subject(tmp_path):
    """If a selected training subject also appears in the test set, the subject
    runner must fail the run rather than train on leaked data (§8)."""
    from signal_aug.data import subject as subj_mod

    pool, test = _leaky_pool_and_test()
    spec = SubjectRunSpec(
        run_id="leak_subj", phase=4, dataset="UCI_HAR",
        augmentation="none", augmentation_params={},
        model="cnn1d_smoke", model_type="cnn1d",
        model_params={"epochs": 1}, seed=0, subject_count=1,
    )
    # force selection of subject 0, which is in test -> guard must trip
    manifest = execute_subject_run(spec, pool, test, runs_dir=tmp_path)
    assert manifest["status"] == "failed"
    log = Path(manifest["log_path"]).read_text()
    assert "subject leak" in log
