"""Subject-level split and leakage-guard tests (spec section 8, Phase 4-5)."""

import numpy as np
import pytest

from signal_aug.data.subject import (
    make_synthetic_subjects,
    select_train_subjects,
    subject_holdout,
    subset_by_subjects,
)


def test_synthetic_subjects_shape():
    data = make_synthetic_subjects(n_subjects=10, samples_per_subject=15)
    assert data.n_subjects() == 10
    assert len(data.y) == 150
    assert data.X.dtype == np.float32


def test_holdout_subjects_are_disjoint():
    data = make_synthetic_subjects(n_subjects=10)
    out = subject_holdout(data, test_subjects=[8, 9], val_subjects=[7])
    train_s = set(out["subjects_train"].tolist())
    assert train_s.isdisjoint({7, 8, 9})
    assert set(out["subjects_test"].tolist()) == {8, 9}
    assert set(out["subjects_val"].tolist()) == {7}


def test_holdout_rejects_overlapping_assignment():
    data = make_synthetic_subjects(n_subjects=6)
    with pytest.raises(ValueError, match="both test and val"):
        subject_holdout(data, test_subjects=[4, 5], val_subjects=[5])


def test_select_train_subjects_deterministic_and_whole():
    data = make_synthetic_subjects(n_subjects=12)
    out = subject_holdout(data, test_subjects=[10, 11])
    a = select_train_subjects(out["subjects_train"], 4, seed=0)
    b = select_train_subjects(out["subjects_train"], 4, seed=0)
    assert a == b
    assert len(a) == 4
    assert set(a) <= set(out["subjects_train"].tolist())


def test_select_train_subjects_nested_growth_is_reasonable():
    data = make_synthetic_subjects(n_subjects=12)
    out = subject_holdout(data, test_subjects=[10, 11])
    counts = [select_train_subjects(out["subjects_train"], n, seed=1) for n in (2, 4, 6)]
    assert [len(c) for c in counts] == [2, 4, 6]


def test_select_too_many_subjects_raises():
    data = make_synthetic_subjects(n_subjects=6)
    out = subject_holdout(data, test_subjects=[5])
    with pytest.raises(ValueError, match="only"):
        select_train_subjects(out["subjects_train"], 10, seed=0)


def test_subset_by_subjects_returns_only_those_subjects():
    data = make_synthetic_subjects(n_subjects=8, samples_per_subject=10)
    out = subject_holdout(data, test_subjects=[6, 7])
    keep = select_train_subjects(out["subjects_train"], 3, seed=2)
    X, y = subset_by_subjects(out, keep)
    assert len(X) == len(y)
    assert len(X) == 3 * 10  # 3 subjects x 10 samples each
