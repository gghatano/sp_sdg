"""WESAD loader unit tests (network-free: channel selection, {1,2,3} filtering,
anti-aliased decimation, windowing/overlap/NaN, subject split, metadata drift and
license recording exercised on tiny synthetic subject dicts / .pkl files). See
src/signal_aug/data/wesad.py."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from signal_aug.data import wesad
from signal_aug.data.subject_datasets import SUBJECT_LOADERS, load_subject_dataset


# --- synthetic subject builders ----------------------------------------------
def _make_raw(label_runs, *, seed=0):
    """Build one WESAD-style subject dict.

    ``label_runs`` is a list of (label_code, n_samples); the label array is their
    concatenation (contiguous runs), and each chest channel is (n, 1). Channel c
    is filled with a c-specific base so channel selection/order is verifiable."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(n, lc, dtype=np.int64) for lc, n in label_runs])
    n = len(labels)
    chest = {}
    for i, key in enumerate(["ECG", "EDA", "EMG", "Resp", "Temp"]):
        chest[key] = (i + 1 + rng.normal(0, 0.01, n)).astype(np.float32).reshape(n, 1)
    chest["ACC"] = rng.normal(0, 1, (n, 3)).astype(np.float32)  # present but excluded
    return {"signal": {"chest": chest}, "label": labels, "subject": "S9"}


def _write_pkl(dir_path, subject, label_runs, *, seed=0):
    raw = _make_raw(label_runs, seed=seed)
    raw["subject"] = f"S{subject}"
    path = dir_path / f"S{subject}.pkl"
    with path.open("wb") as f:
        pickle.dump(raw, f)
    return path


# --- channel resolution -------------------------------------------------------
def test_resolve_chest_channels_default_is_5ch_physiology():
    keys = wesad.resolve_chest_channels(wesad.DEFAULT_CHANNELS)
    assert keys == ["ECG", "EDA", "EMG", "Resp", "Temp"]
    assert len(keys) == 5


def test_resolve_chest_channels_rejects_acc_and_wrist():
    with pytest.raises(ValueError, match="unknown WESAD chest channel"):
        wesad.resolve_chest_channels(["ACC"])       # motion (HAR-like) excluded
    with pytest.raises(ValueError, match="unknown WESAD chest channel"):
        wesad.resolve_chest_channels(["BVP"])       # wrist E4 channel excluded


def test_downsample_factor():
    assert wesad.downsample_factor(70) == 10
    assert wesad.downsample_factor(100) == 7
    assert wesad.downsample_factor(64) == 11
    with pytest.raises(ValueError):
        wesad.downsample_factor(1400)  # faster than native 700 Hz -> factor < 1


# --- parsing / label filtering ------------------------------------------------
def test_parse_pkl_selects_channels_in_order_and_keeps_raw_labels():
    raw = _make_raw([(1, 20), (2, 20)])
    values, labels = wesad.parse_pkl(raw, wesad.DEFAULT_CHANNELS)
    assert values.shape == (40, 5)
    # channel order ECG,EDA,EMG,RESP,TEMP -> bases ~1,2,3,4,5
    assert np.allclose(values.mean(axis=0), [1, 2, 3, 4, 5], atol=0.1)
    assert set(labels.tolist()) == {1, 2}  # raw codes, not yet remapped


def test_build_arrays_keeps_only_123_and_remaps(tmp_path):
    # labels 0 (transient), 1, 2, 3, 4 (meditation) -> keep {1,2,3} -> {0,1,2}
    _write_pkl(tmp_path, 9, [(0, 30), (1, 40), (2, 40), (3, 40), (4, 30)])
    values, labels, subjects = wesad.build_arrays(tmp_path)
    assert values.shape[1] == 5                    # 5 chest physiology channels
    assert set(labels.tolist()) == {0, 1, 2}       # baseline/stress/amusement remapped
    assert len(labels) == 120                      # 40+40+40, transient/meditation dropped
    assert set(subjects.tolist()) == {9}


def test_build_arrays_honours_exclude_subjects(tmp_path):
    _write_pkl(tmp_path, 9, [(1, 40)])
    _write_pkl(tmp_path, 17, [(1, 40)])
    _, _, subjects = wesad.build_arrays(tmp_path, exclude_subjects={17})
    assert set(subjects.tolist()) == {9}


def test_subject_id_parsing_rejects_bad_stem(tmp_path):
    with pytest.raises(ValueError, match="unexpected WESAD subject file stem"):
        wesad._subject_id("subject9")


# --- decimation / anti-aliasing ----------------------------------------------
def test_decimate_block_is_antialiased_not_naive_stride():
    fs, factor = 700, 10
    t = np.arange(7000) / fs
    # a strong tone at 200 Hz, far above the post-decimation Nyquist (35 Hz):
    # naive striding aliases it back into band; anti-aliased decimation removes it.
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32).reshape(-1, 1)
    deci = wesad._decimate_block(tone, factor)
    stride = tone[::factor]
    assert deci.shape[0] == stride.shape[0]  # same output length
    deci_rms = float(np.sqrt((deci**2).mean()))
    stride_rms = float(np.sqrt((stride**2).mean()))
    assert deci_rms < 0.1 * stride_rms       # out-of-band tone strongly attenuated


def test_decimate_block_noop_when_factor_1():
    block = np.random.default_rng(0).normal(0, 1, (100, 5)).astype(np.float32)
    out = wesad._decimate_block(block, 1)
    assert np.array_equal(out, block)


def test_decimate_block_returns_empty_for_tiny_block():
    tiny = np.ones((10, 5), dtype=np.float32)
    assert wesad._decimate_block(tiny, 10).shape[0] == 0


# --- windowing ----------------------------------------------------------------
def test_make_windows_nonoverlap_shape_and_znorm():
    # factor=1 (no decimation) so shapes are exact: 300 samples / window 100 = 3
    values = np.random.default_rng(0).normal(5, 2, size=(300, 5)).astype(np.float32)
    labels = np.zeros(300, dtype=np.int64)
    subjects = np.full(300, 9, dtype=np.int64)
    X, y, subj = wesad.make_windows(values, labels, subjects, window=100, downsample_factor=1)
    assert X.shape == (3, 5, 100)
    assert set(y.tolist()) == {0} and set(subj.tolist()) == {9}
    assert np.allclose(X[0].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(X[0].std(axis=1), 1, atol=1e-2)


def test_make_windows_overlap_adds_windows():
    values = np.random.default_rng(1).normal(0, 1, size=(300, 5)).astype(np.float32)
    labels = np.zeros(300, dtype=np.int64)
    subjects = np.full(300, 9, dtype=np.int64)
    non = wesad.make_windows(values, labels, subjects, window=100, downsample_factor=1, overlap=0.0)[0]
    ov = wesad.make_windows(values, labels, subjects, window=100, downsample_factor=1, overlap=0.5)[0]
    assert non.shape[0] == 3           # step 100 -> 3
    assert ov.shape[0] == 5            # step 50 -> starts 0,50,100,150,200 -> 5


def test_make_windows_drops_nan_windows():
    clean = np.random.default_rng(1).normal(0, 1, size=(200, 5)).astype(np.float32)
    dirty = clean.copy()
    dirty[:, 2] = np.nan  # an EMG dropout across the whole second block
    values = np.concatenate([clean, dirty])
    labels = np.concatenate([np.zeros(200), np.ones(200)]).astype(np.int64)
    subjects = np.full(400, 9, dtype=np.int64)
    X, y, _ = wesad.make_windows(values, labels, subjects, window=200, downsample_factor=1)
    assert X.shape[0] == 1           # clean block -> 1 window; dirty dropped
    assert set(y.tolist()) == {0}


def test_make_windows_never_mixes_subject_or_label():
    blocks, labels, subjects = [], [], []
    for subj in (9, 17):
        for lab in (0, 1):
            blocks.append(np.random.default_rng(subj + lab).normal(0, 1, size=(200, 5)).astype(np.float32))
            labels.append(np.full(200, lab))
            subjects.append(np.full(200, subj))
    X, y, subj_out = wesad.make_windows(
        np.concatenate(blocks), np.concatenate(labels).astype(np.int64),
        np.concatenate(subjects).astype(np.int64), window=200, downsample_factor=1,
    )
    assert X.shape[0] == 2 * 2 * 1  # (2 subj x 2 label) x 1 window each
    assert set(y.tolist()) == {0, 1} and set(subj_out.tolist()) == {9, 17}


def test_make_windows_rejects_unknown_normalize_and_bad_overlap():
    values = np.zeros((200, 5), dtype=np.float32)
    labels = np.zeros(200, dtype=np.int64)
    with pytest.raises(ValueError, match="unknown normalize"):
        wesad.make_windows(values, labels, labels, window=100, normalize="bogus")
    with pytest.raises(ValueError, match="overlap must be"):
        wesad.make_windows(values, labels, labels, window=100, downsample_factor=1, overlap=1.0)


# --- subject split ------------------------------------------------------------
def test_split_subjects_disjoint_deterministic_and_config():
    subj = np.repeat(np.arange(2, 18), 5)  # 16 ids ~ WESAD S2..S17 range
    pool_a, test_a = wesad._split_subjects(subj)
    pool_b, test_b = wesad._split_subjects(subj)
    assert (pool_a, test_a) == (pool_b, test_b)          # deterministic
    assert len(test_a) == wesad.N_TEST_SUBJECTS == 5
    assert not (set(pool_a) & set(test_a))              # disjoint
    assert set(pool_a) | set(test_a) == set(range(2, 18))
    _, test_seed1 = wesad._split_subjects(subj, split_seed=1)
    assert set(test_a) != set(test_seed1)               # config seed flows in


def test_split_subjects_raises_when_pool_empty():
    subj = np.repeat(np.array([2, 3, 4, 5, 6]), 5)
    with pytest.raises(ValueError, match="no pool subjects"):
        wesad._split_subjects(subj, n_test_subjects=5)


def test_per_subject_class_support_counts_and_absences():
    y = np.array([0, 0, 1, 2], dtype=np.int64)
    subj = np.array([9, 9, 9, 17], dtype=np.int64)
    support = wesad.per_subject_class_support(y, subj)
    assert support[9] == {"baseline": 2, "stress": 1, "amusement": 0}
    assert support[17] == {"baseline": 0, "stress": 0, "amusement": 1}


# --- metadata -----------------------------------------------------------------
def _tiny_pkls(tmp_path, subjects=(9, 17, 3, 4, 5, 6)):
    for i, s in enumerate(subjects):
        _write_pkl(tmp_path, s, [(1, 300), (2, 300), (3, 300)], seed=s + i)
    return tmp_path


def test_record_metadata_records_license_and_support(tmp_path):
    pkl_dir = _tiny_pkls(tmp_path)
    values, labels, subjects = wesad.build_arrays(pkl_dir)
    X, y, subj = wesad.make_windows(values, labels, subjects, window=100, downsample_factor=1)
    pool_s, test_s = wesad._split_subjects(subj, n_test_subjects=2)
    meta = wesad._record_metadata(
        tmp_path, X, y, subj, pool_s, test_s,
        window=100, split_seed=0, n_test_subjects=2, normalize="per_window_z",
        downsample_hz=70, overlap=0.0, channels=wesad.DEFAULT_CHANNELS,
    )
    saved = json.loads((tmp_path / "wesad.json").read_text())
    assert saved["n_classes"] == 3 and saved["class_names"] == ["baseline", "stress", "amusement"]
    assert saved["channels"] == wesad.DEFAULT_CHANNELS and saved["modality"] == "chest"
    assert "non-commercial" in saved["license"].lower() and "not cc by" in saved["license"].lower()
    assert saved["doi"] == "10.1145/3242969.3242985"
    assert "not granted" in saved["redistribution"]
    assert saved["pool_windows_checksum"] and saved["test_windows_checksum"]
    assert "per_subject_class_support" in saved
    assert meta["n_pool_windows"] > 0


def test_record_metadata_detects_drift(tmp_path):
    pkl_dir = _tiny_pkls(tmp_path)
    values, labels, subjects = wesad.build_arrays(pkl_dir)

    def record(window):
        X, y, subj = wesad.make_windows(values, labels, subjects, window=window, downsample_factor=1)
        pool_s, test_s = wesad._split_subjects(subj, n_test_subjects=2)
        wesad._record_metadata(
            tmp_path, X, y, subj, pool_s, test_s,
            window=window, split_seed=0, n_test_subjects=2, normalize="per_window_z",
            downsample_hz=70, overlap=0.0, channels=wesad.DEFAULT_CHANNELS,
        )

    record(100)  # first write
    record(100)  # identical -> no error
    with pytest.raises(ValueError, match="metadata drift"):
        record(150)  # different window set -> drift caught


# --- end-to-end / dispatch ----------------------------------------------------
def test_load_wesad_end_to_end(tmp_path, monkeypatch):
    pkl_dir = tmp_path / "pkls"
    pkl_dir.mkdir()
    _tiny_pkls(pkl_dir, subjects=(9, 17, 3, 4, 5, 6, 7))
    monkeypatch.setattr(wesad, "_ensure_extracted", lambda *a, **k: pkl_dir)
    # downsample_hz=700 -> factor 1 (no decimation) keeps the tiny blocks usable
    pool, test = wesad.load_wesad(
        metadata_dir=tmp_path, window=100, downsample_hz=700, n_test_subjects=2,
    )
    assert set(pool.unique_subjects().tolist()).isdisjoint(test.unique_subjects().tolist())
    assert test.n_subjects() == 2 and pool.n_subjects() == 5
    assert pool.X.shape[1] == 5           # 5 chest channels
    assert set(pool.y.tolist()) <= {0, 1, 2}
    meta = json.loads((tmp_path / "wesad.json").read_text())
    assert meta["dataset"] == "WESAD" and meta["n_classes"] == 3


def test_load_wesad_rejects_non_chest_modality(tmp_path, monkeypatch):
    monkeypatch.setattr(wesad, "_ensure_extracted", lambda *a, **k: tmp_path)
    with pytest.raises(ValueError, match="only implements modality"):
        wesad.load_wesad(metadata_dir=tmp_path, modality="wrist")


def test_dispatch_registry_includes_wesad():
    assert set(SUBJECT_LOADERS) >= {"UCI_HAR", "WISDM", "PAMAP2", "WESAD"}
    with pytest.raises(ValueError, match="unknown subject dataset"):
        load_subject_dataset("NOT_A_DATASET")
