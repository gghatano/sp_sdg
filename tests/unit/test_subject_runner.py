"""Subject-count grid expansion and pre-registration guard (Phase 4)."""

import numpy as np
import pytest

from signal_aug.data.subject import SubjectSplits, windows_checksum
from signal_aug.experiments.subject_runner import (
    SubjectRunSpec,
    execute_subject_run,
    expand_subject_grid,
    run_subject_experiment,
)
from signal_aug.models import factory

EXP = {
    "phase": 4,
    "name": "subject_count",
    "dataset": "UCI_HAR",
    "model": "cnn1d",
    "subject_counts": [3, 6],
    "repeats": 2,
    "augmentations": ["none", "mixup"],
    "pre_registered": True,
}
AUG = {"augmentations": {"none": {"method": "none"}, "mixup": {"method": "mixup", "params": {"ratio": 1.0}}}}
MODEL = {"models": {"cnn1d": {"type": "cnn1d", "params": {"epochs": 5}}}}


def test_subject_grid_size_and_ids():
    runs = expand_subject_grid(EXP, AUG, MODEL)
    assert len(runs) == 2 * 2 * 2  # counts x repeats x augs
    ids = {r.run_id for r in runs}
    assert "subject_count_UCI_HAR_n3_none_cnn1d_r0" in ids
    assert "subject_count_UCI_HAR_n6_mixup_cnn1d_r1" in ids
    assert all(r.subject_count in (3, 6) for r in runs)


def test_subject_grid_carries_params():
    runs = expand_subject_grid(EXP, AUG, MODEL)
    mixup = next(r for r in runs if r.augmentation == "mixup")
    assert mixup.augmentation_params == {"ratio": 1.0}
    assert mixup.model_params == {"epochs": 5}


def test_run_rejects_unregistered_config(tmp_path):
    import yaml
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(dict(EXP, pre_registered=False)))
    with pytest.raises(ValueError, match="pre-registered"):
        run_subject_experiment(cfg, runs_dir=tmp_path / "runs")


def test_subject_grid_carries_dataset_params():
    exp = dict(EXP, dataset="WISDM", dataset_params={"window": 200, "normalize": "none"})
    runs = expand_subject_grid(exp, AUG, MODEL)
    assert all(r.dataset_params == {"window": 200, "normalize": "none"} for r in runs)
    # default when the config omits the block (backward compatible)
    assert all(r.dataset_params == {} for r in expand_subject_grid(EXP, AUG, MODEL))


class _SpyModel:
    def __init__(self, seed=0):
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


def _pool_and_test():
    rng = np.random.default_rng(0)
    n, length = 40, 16
    X = rng.normal(0, 1, size=(n, 1, length)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    pool = SubjectSplits("pool", X[:24], y[:24], np.array([i % 4 for i in range(24)]))
    test = SubjectSplits("test", X[24:], y[24:], np.array([100 + i % 3 for i in range(16)]))
    return pool, test


def test_manifest_records_real_checksum_and_dataset_params(tmp_path, monkeypatch):
    monkeypatch.setitem(factory.REGISTRY, "spy", lambda params, seed: _SpyModel(seed=seed))
    pool, test = _pool_and_test()
    spec = SubjectRunSpec(
        run_id="wisdm_ck", phase=6, dataset="WISDM",
        augmentation="none", augmentation_params={},
        model="spy", model_type="spy", model_params={}, seed=0, subject_count=2,
        dataset_params={"window": 200, "split_seed": 0},
    )
    manifest = execute_subject_run(spec, pool, test, runs_dir=tmp_path)
    assert manifest["status"] == "completed"
    # dataset_checksum is a real content hash, not the old fixed "wisdm_pool"
    assert manifest["dataset_checksum"] == f"wisdm:{windows_checksum(pool.X, test.X)}"
    assert manifest["dataset_checksum"] != "wisdm_pool"
    assert manifest["dataset_params"] == {"window": 200, "split_seed": 0}


def test_manifest_checksum_changes_with_window_set(tmp_path, monkeypatch):
    monkeypatch.setitem(factory.REGISTRY, "spy", lambda params, seed: _SpyModel(seed=seed))
    pool, test = _pool_and_test()
    base_spec = dict(
        phase=6, dataset="WISDM", augmentation="none", augmentation_params={},
        model="spy", model_type="spy", model_params={}, seed=0, subject_count=2,
    )
    m1 = execute_subject_run(SubjectRunSpec(run_id="ck1", **base_spec), pool, test, runs_dir=tmp_path)
    # a changed window set (e.g. different windowing) must change the checksum
    pool2 = SubjectSplits("pool", pool.X[:, :, :8].copy(), pool.y, pool.subjects)
    m2 = execute_subject_run(SubjectRunSpec(run_id="ck2", **base_spec), pool2, test, runs_dir=tmp_path)
    assert m1["dataset_checksum"] != m2["dataset_checksum"]
