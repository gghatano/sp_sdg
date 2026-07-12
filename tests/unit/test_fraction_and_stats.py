"""Phase 2 additions: stratified subsampling, fraction-aware grids, and the
Wilcoxon comparison against the none baseline."""

import numpy as np
import pytest

from signal_aug.data.loader import load_synthetic, stratified_subsample
from signal_aug.evaluation.stats import paired_deltas, wilcoxon_vs_none
from signal_aug.experiments.runner import expand_grid

SPEC = {"n_train": 40, "n_test": 20, "length": 64, "n_channels": 1, "n_classes": 2}


def test_subsample_stratified_and_sized():
    data = load_synthetic(SPEC)
    X_sub, y_sub = stratified_subsample(data.X_train, data.y_train, 0.5, seed=0)
    assert len(X_sub) == len(y_sub)
    assert abs(len(X_sub) - 20) <= 1
    assert set(np.unique(y_sub)) == set(np.unique(data.y_train))


def test_subsample_deterministic_and_seed_sensitive():
    data = load_synthetic(SPEC)
    a = stratified_subsample(data.X_train, data.y_train, 0.25, seed=1)
    b = stratified_subsample(data.X_train, data.y_train, 0.25, seed=1)
    c = stratified_subsample(data.X_train, data.y_train, 0.25, seed=2)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_subsample_full_fraction_is_identity():
    data = load_synthetic(SPEC)
    X_sub, y_sub = stratified_subsample(data.X_train, data.y_train, 1.0, seed=0)
    np.testing.assert_array_equal(X_sub, data.X_train)
    np.testing.assert_array_equal(y_sub, data.y_train)


def test_subsample_rejects_invalid_fraction():
    data = load_synthetic(SPEC)
    with pytest.raises(ValueError):
        stratified_subsample(data.X_train, data.y_train, 0.0, seed=0)


GRID_CFG = {
    "phase": 2,
    "name": "p2test",
    "datasets": ["synthetic"],
    "augmentations": ["none", "jitter"],
    "models": ["cnn1d_smoke"],
    "seeds": [0],
}
AUG_CFG = {"augmentations": {"none": {"method": "none"}, "jitter": {"method": "jitter", "params": {}}}}
MODEL_CFG = {"models": {"cnn1d_smoke": {"type": "cnn1d", "params": {}}}}


def test_grid_without_fractions_keeps_run_ids_stable():
    runs = expand_grid(GRID_CFG, AUG_CFG, MODEL_CFG)
    assert [r.run_id for r in runs] == [
        "p2test_synthetic_none_cnn1d_smoke_s0",
        "p2test_synthetic_jitter_cnn1d_smoke_s0",
    ]
    assert all(r.train_fraction == 1.0 for r in runs)


def test_grid_with_fractions_tags_run_ids():
    cfg = dict(GRID_CFG, train_fractions=[0.25, 1.0])
    runs = expand_grid(cfg, AUG_CFG, MODEL_CFG)
    ids = [r.run_id for r in runs]
    assert "p2test_synthetic_f25_none_cnn1d_smoke_s0" in ids
    assert "p2test_synthetic_none_cnn1d_smoke_s0" in ids  # 1.0 keeps the bare id
    assert len(runs) == 4


def _rows():
    rows = []
    for seed in range(6):
        rows.append({"dataset": "d", "model": "m", "seed": seed, "train_fraction": 1.0,
                     "augmentation": "none", "status": "completed", "accuracy": 0.80})
        rows.append({"dataset": "d", "model": "m", "seed": seed, "train_fraction": 1.0,
                     "augmentation": "jitter", "status": "completed", "accuracy": 0.85})
    return rows


def test_paired_deltas_matches_pairs():
    deltas = paired_deltas(_rows(), "jitter", "m")
    assert len(deltas) == 6
    assert all(abs(d - 0.05) < 1e-9 for d in deltas)


def test_wilcoxon_detects_consistent_improvement():
    results = wilcoxon_vs_none(_rows())
    assert len(results) == 1
    r = results[0]
    assert r["augmentation"] == "jitter" and r["n_pairs"] == 6
    assert r["mean_delta"] == 0.05
    assert r["p_value"] is not None and r["p_value"] < 0.05


def test_wilcoxon_skips_underpowered_groups():
    rows = _rows()[:4]  # only 2 pairs
    assert wilcoxon_vs_none(rows) == []


def test_wilcoxon_all_zero_deltas_returns_none_pvalue():
    rows = []
    for seed in range(6):
        for aug in ("none", "jitter"):
            rows.append({"dataset": "d", "model": "m", "seed": seed, "train_fraction": 1.0,
                         "augmentation": aug, "status": "completed", "accuracy": 0.8})
    res = wilcoxon_vs_none(rows)  # jitter == none everywhere -> all-zero diffs
    jitter = next(r for r in res if r["augmentation"] == "jitter")
    assert jitter["p_value"] is None
    assert jitter["mean_delta"] == 0.0


def test_wilcoxon_mixed_signs_still_computes():
    rows = []
    deltas = [0.05, -0.02, 0.03, -0.01, 0.04, 0.06]
    for seed, d in enumerate(deltas):
        rows.append({"dataset": "d", "model": "m", "seed": seed, "train_fraction": 1.0,
                     "augmentation": "none", "status": "completed", "accuracy": 0.80})
        rows.append({"dataset": "d", "model": "m", "seed": seed, "train_fraction": 1.0,
                     "augmentation": "jitter", "status": "completed", "accuracy": 0.80 + d})
    res = wilcoxon_vs_none(rows)
    jitter = next(r for r in res if r["augmentation"] == "jitter")
    assert jitter["p_value"] is not None
    assert jitter["n_pairs"] == 6
