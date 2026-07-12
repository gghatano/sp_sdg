"""Synthetic loader, znorm, checksums, and train/val split checks (spec 3.6)."""

import numpy as np

from signal_aug.data.loader import load_synthetic, train_val_split

SPEC = {"n_train": 40, "n_test": 20, "length": 64, "n_channels": 1, "n_classes": 2}


def test_synthetic_shapes_and_dtypes():
    data = load_synthetic(SPEC)
    assert data.X_train.shape == (40, 1, 64)
    assert data.X_test.shape == (20, 1, 64)
    assert data.X_train.dtype == np.float32
    assert data.y_train.dtype == np.int64


def test_synthetic_deterministic_checksums():
    a, b = load_synthetic(SPEC), load_synthetic(SPEC)
    assert a.dataset_checksum == b.dataset_checksum
    assert a.split_checksum == b.split_checksum


def test_no_train_test_duplicates():
    data = load_synthetic(SPEC)
    train_rows = {row.tobytes() for row in data.X_train.reshape(len(data.X_train), -1)}
    test_rows = {row.tobytes() for row in data.X_test.reshape(len(data.X_test), -1)}
    assert not train_rows & test_rows


def test_train_val_split_no_overlap_and_stratified():
    data = load_synthetic(SPEC)
    X_tr, y_tr, X_val, y_val = train_val_split(data.X_train, data.y_train, 0.25, seed=0)
    assert len(X_tr) + len(X_val) == len(data.X_train)
    tr_rows = {row.tobytes() for row in X_tr.reshape(len(X_tr), -1)}
    val_rows = {row.tobytes() for row in X_val.reshape(len(X_val), -1)}
    assert not tr_rows & val_rows
    assert set(np.unique(y_val)) == set(np.unique(y_tr))


def test_train_val_split_seed_reproducible():
    data = load_synthetic(SPEC)
    a = train_val_split(data.X_train, data.y_train, 0.25, seed=3)
    b = train_val_split(data.X_train, data.y_train, 0.25, seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[3], b[3])
