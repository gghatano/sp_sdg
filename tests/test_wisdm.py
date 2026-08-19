"""WISDM v1.1 loader unit tests (network-free: parsing/windowing/split logic
exercised on a tiny synthetic raw string). See src/signal_aug/data/wisdm.py."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from signal_aug.data import wisdm
from signal_aug.data.subject_datasets import SUBJECT_LOADERS, load_subject_dataset


def _synthetic_raw(users=(1, 2, 3), per_block=250) -> str:
    """Build a raw-format string: 'user,activity,ts,x,y,z;' records."""
    rng = np.random.default_rng(0)
    recs = []
    ts = 0
    for u in users:
        for act in ("Walking", "Jogging"):
            for _ in range(per_block):
                x, y, z = rng.normal(0, 5, 3)
                recs.append(f"{u},{act},{ts},{x:.4f},{y:.4f},{z:.4f};")
                ts += 50_000_000
    # inject malformed rows that must be skipped
    recs.append("1,Walking,123,onlyfour,5;")            # wrong field count
    recs.append("2,Flying,123,1.0,2.0,3.0;")            # unknown activity
    recs.append("x,Walking,123,1.0,2.0,3.0;")           # non-int user
    recs.append("")                                     # empty
    return "\n".join(recs)


def test_parse_raw_skips_malformed():
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw())
    assert xyz.dtype == np.float32 and xyz.shape[1] == 3
    assert labels.dtype == np.int64 and subjects.dtype == np.int64
    # only Walking/Jogging survive; unknown activity / bad user dropped
    assert set(labels.tolist()) <= {0, 1}
    assert set(subjects.tolist()) == {1, 2, 3}
    # 3 users x 2 activities x 250 valid samples
    assert len(labels) == 3 * 2 * 250


def test_make_windows_shape_and_single_label():
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=250))
    X, y, subj = wisdm.make_windows(xyz, labels, subjects, window=200)
    assert X.shape[1:] == (3, 200)
    assert X.dtype == np.float32
    # 250 samples per (subject, activity) block -> exactly 1 non-overlap window
    assert len(y) == 3 * 2 * 1
    # per-window z-norm: each channel ~ mean 0, std 1
    assert np.allclose(X[0].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(X[0].std(axis=1), 1, atol=1e-2)


def test_windows_never_mix_subject_or_activity():
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=450))
    X, y, subj = wisdm.make_windows(xyz, labels, subjects, window=200)
    # 450 samples per block -> 2 windows per (subject, activity); labels stay pure
    assert len(y) == 3 * 2 * 2
    assert set(subj.tolist()) == {1, 2, 3}


def test_split_subjects_disjoint_and_deterministic():
    subj = np.repeat(np.arange(1, 37), 3)
    pool_a, test_a = wisdm._split_subjects(subj)
    pool_b, test_b = wisdm._split_subjects(subj)
    assert (pool_a, test_a) == (pool_b, test_b)          # deterministic
    assert len(test_a) == wisdm.N_TEST_SUBJECTS
    assert len(pool_a) == 36 - wisdm.N_TEST_SUBJECTS
    assert not (set(pool_a) & set(test_a))               # disjoint
    assert set(pool_a) | set(test_a) == set(range(1, 37))


def test_split_subjects_honours_config_params():
    subj = np.repeat(np.arange(1, 37), 3)
    pool, test = wisdm._split_subjects(subj, n_test_subjects=6, split_seed=1)
    assert len(test) == 6 and len(pool) == 30
    # a different seed yields a different held-out set (config actually flows in)
    _, test_seed0 = wisdm._split_subjects(subj, n_test_subjects=6, split_seed=0)
    assert set(test) != set(test_seed0)


def test_make_windows_normalize_none_keeps_raw_scale():
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=250))
    X, _, _ = wisdm.make_windows(xyz, labels, subjects, window=200, normalize="none")
    # raw accel is on ~sigma-5 scale, so at least one window channel std >> 1
    assert X.std() > 2.0
    # and it differs from the z-normed windows (config knob has an effect)
    Xz, _, _ = wisdm.make_windows(xyz, labels, subjects, window=200, normalize="per_window_z")
    assert not np.allclose(X, Xz)


def test_make_windows_rejects_unknown_normalize():
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=250))
    with pytest.raises(ValueError, match="unknown normalize"):
        wisdm.make_windows(xyz, labels, subjects, window=200, normalize="bogus")


def test_record_metadata_records_config_and_checksums(tmp_path):
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=450))
    X, _, subj = wisdm.make_windows(xyz, labels, subjects, window=200)
    pool_s, test_s = wisdm._split_subjects(subj, n_test_subjects=1, split_seed=0)
    wisdm._record_metadata(
        tmp_path, X, labels, subj, pool_s, test_s,
        window=200, split_seed=0, n_test_subjects=1, normalize="per_window_z",
    )
    meta = json.loads((tmp_path / "wisdm.json").read_text())
    assert meta["window_length"] == 200
    assert meta["split_seed"] == 0
    assert meta["n_test_subjects"] == 1
    assert meta["normalize"] == "per_window_z"
    assert meta["pool_windows_checksum"] and meta["test_windows_checksum"]


def test_record_metadata_detects_window_drift(tmp_path):
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=450))
    subj_all = wisdm.make_windows(xyz, labels, subjects, window=200)[2]
    pool_s, test_s = wisdm._split_subjects(subj_all, n_test_subjects=1, split_seed=0)

    def record(window):
        X, _, subj = wisdm.make_windows(xyz, labels, subjects, window=window)
        wisdm._record_metadata(
            tmp_path, X, labels, subj, pool_s, test_s,
            window=window, split_seed=0, n_test_subjects=1, normalize="per_window_z",
        )

    record(200)                       # first write
    record(200)                       # identical -> no error, left in place
    with pytest.raises(ValueError, match="metadata drift"):
        record(225)                   # different window set -> drift caught


def test_record_metadata_migrates_precheckum_file(tmp_path):
    # An old-schema metadata file (no checksum) must be overwritten, not error.
    (tmp_path / "wisdm.json").write_text(json.dumps({"dataset": "WISDM", "window_length": 200}))
    xyz, labels, subjects = wisdm.parse_raw(_synthetic_raw(per_block=450))
    X, _, subj = wisdm.make_windows(xyz, labels, subjects, window=200)
    pool_s, test_s = wisdm._split_subjects(subj, n_test_subjects=1, split_seed=0)
    wisdm._record_metadata(
        tmp_path, X, labels, subj, pool_s, test_s,
        window=200, split_seed=0, n_test_subjects=1, normalize="per_window_z",
    )
    meta = json.loads((tmp_path / "wisdm.json").read_text())
    assert "pool_windows_checksum" in meta  # migrated to the checksum schema


def test_download_verifies_tar_checksum(tmp_path):
    tar_path = tmp_path / "wisdm.tar.gz"
    tar_path.write_bytes(b"\x00" * 1_100_000)   # >1MB so no download is attempted
    good = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    wisdm._download(tar_path, expected_sha256=good)  # matches -> no error
    with pytest.raises(ValueError, match="tar checksum mismatch"):
        wisdm._download(tar_path, expected_sha256="deadbeef")


def test_dispatch_registry_and_unknown_key():
    assert set(SUBJECT_LOADERS) >= {"UCI_HAR", "WISDM"}
    with pytest.raises(ValueError, match="unknown subject dataset"):
        load_subject_dataset("NOT_A_DATASET")
