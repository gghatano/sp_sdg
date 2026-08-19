"""PAMAP2 loader unit tests (network-free: parsing/windowing/split/channel and
metadata logic exercised on tiny synthetic .dat text). See
src/signal_aug/data/pamap2.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from signal_aug.data import pamap2
from signal_aug.data.subject_datasets import SUBJECT_LOADERS, load_subject_dataset


def _make_rows(activity: int, n: int, *, seed: int = 0, nan_col: int | None = None) -> np.ndarray:
    """Build (n, 54) rows for one activity. Column c is filled with c + noise so
    channel selection is verifiable; HR (col 2) is NaN like the real data."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 54), dtype=np.float64)
    for c in range(54):
        data[:, c] = c + rng.normal(0, 0.01, n)
    data[:, 1] = activity            # activityID
    data[:, 2] = np.nan              # heart rate mostly missing
    if nan_col is not None:
        data[:, nan_col] = np.nan    # simulate a wireless dropout in one channel
    return data


def _to_text(data: np.ndarray) -> str:
    return "\n".join(" ".join(f"{v:.6f}" for v in row) for row in data)


def test_resolve_channel_columns_default_is_accel16g_9ch():
    cols = pamap2.resolve_channel_columns(pamap2.DEFAULT_CHANNELS)
    # hand base 3 (+1..3), chest base 20 (+1..3), ankle base 37 (+1..3)
    assert cols == [4, 5, 6, 21, 22, 23, 38, 39, 40]
    assert len(cols) == 9


def test_resolve_channel_columns_rejects_unknown():
    with pytest.raises(ValueError, match="unknown PAMAP2 channel"):
        pamap2.resolve_channel_columns(["hand_orientation_0"])  # bad sensor name
    with pytest.raises(ValueError, match="unknown PAMAP2 channel"):
        pamap2.resolve_channel_columns(["wrist_acc16_x"])       # bad IMU location


def test_downsample_factor():
    assert pamap2.downsample_factor(33) == 3
    assert pamap2.downsample_factor(34) == 3
    assert pamap2.downsample_factor(50) == 2
    with pytest.raises(ValueError):
        pamap2.downsample_factor(200)  # faster than native -> factor < 1


def test_parse_dat_preserves_nan_and_shape():
    data = _make_rows(activity=4, n=10)
    parsed, activity = pamap2.parse_dat(_to_text(data))
    assert parsed.shape == (10, 54)
    assert parsed.dtype == np.float32
    assert np.isnan(parsed[:, 2]).all()          # HR column NaN preserved
    assert set(activity.tolist()) == {4}
    assert np.allclose(parsed[:, 40], 40, atol=0.1)  # ankle_acc16_z ~ col index


def test_build_arrays_drops_transient_and_optional(tmp_path):
    protocol = tmp_path / "Protocol"
    protocol.mkdir()
    # subject 101: transient(0) + protocol walking(4) + optional computer_work(10)
    rows = np.concatenate([
        _make_rows(0, 30),    # transient -> dropped
        _make_rows(4, 40),    # protocol walking -> label 3
        _make_rows(10, 25),   # optional -> dropped (not in protocol subset)
    ])
    (protocol / "subject101.dat").write_text(_to_text(rows))
    values, labels, subjects = pamap2.build_arrays(protocol)
    assert values.shape[1] == 9                   # 9 accel channels selected
    assert set(labels.tolist()) == {3}            # only walking survives
    assert set(subjects.tolist()) == {101}
    assert len(labels) == 40


def test_build_arrays_honours_exclude_subjects(tmp_path):
    protocol = tmp_path / "Protocol"
    protocol.mkdir()
    (protocol / "subject101.dat").write_text(_to_text(_make_rows(4, 40)))
    (protocol / "subject108.dat").write_text(_to_text(_make_rows(4, 40)))
    _, _, subjects = pamap2.build_arrays(protocol, exclude_subjects={108})
    assert set(subjects.tolist()) == {101}


def test_make_windows_shape_downsample_and_znorm():
    # one (subject, activity) block: 1008 raw -> /3 = 336 -> /168 = 2 windows
    values = np.random.default_rng(0).normal(0, 5, size=(1008, 9)).astype(np.float32)
    labels = np.full(1008, 3, dtype=np.int64)
    subjects = np.full(1008, 101, dtype=np.int64)
    X, y, subj = pamap2.make_windows(values, labels, subjects, window=168, downsample_factor=3)
    assert X.shape == (2, 9, 168)
    assert set(y.tolist()) == {3} and set(subj.tolist()) == {101}
    # per-window z-norm: each channel ~ mean 0, std 1
    assert np.allclose(X[0].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(X[0].std(axis=1), 1, atol=1e-2)


def test_make_windows_stride_actually_decimates():
    values = np.arange(168 * 3, dtype=np.float32)[:, None].repeat(9, axis=1)
    labels = np.zeros(168 * 3, dtype=np.int64)
    subjects = np.zeros(168 * 3, dtype=np.int64)
    # factor 3 keeps every 3rd row -> exactly one 168-window; factor 1 -> 3 windows
    X3, _, _ = pamap2.make_windows(values, labels, subjects, window=168, downsample_factor=3, normalize="none")
    X1, _, _ = pamap2.make_windows(values, labels, subjects, window=168, downsample_factor=1, normalize="none")
    assert X3.shape[0] == 1 and X1.shape[0] == 3
    # decimated window is rows 0,3,6,... -> first three values 0,3,6
    assert list(X3[0, 0, :3]) == [0.0, 3.0, 6.0]


def test_make_windows_drops_nan_windows():
    clean = np.random.default_rng(1).normal(0, 1, size=(504, 9)).astype(np.float32)
    dirty = clean.copy()
    dirty[:, 4] = np.nan  # a channel dropout across the whole second block
    values = np.concatenate([clean, dirty])
    labels = np.concatenate([np.full(504, 3), np.full(504, 5)]).astype(np.int64)
    subjects = np.full(1008, 101, dtype=np.int64)
    X, y, _ = pamap2.make_windows(values, labels, subjects, window=168, downsample_factor=3)
    # clean block -> 504/3/168 = 1 window; dirty block dropped entirely
    assert X.shape[0] == 1
    assert set(y.tolist()) == {3}


def test_make_windows_never_mixes_subject_or_activity():
    blocks, labels, subjects = [], [], []
    for subj in (101, 102):
        for act in (3, 5):
            blocks.append(np.random.default_rng(subj + act).normal(0, 1, size=(504, 9)).astype(np.float32))
            labels.append(np.full(504, act))
            subjects.append(np.full(504, subj))
    X, y, subj_out = pamap2.make_windows(
        np.concatenate(blocks), np.concatenate(labels).astype(np.int64),
        np.concatenate(subjects).astype(np.int64), window=168, downsample_factor=3,
    )
    assert X.shape[0] == 2 * 2 * 1  # (2 subj x 2 act) x 1 window each
    assert set(y.tolist()) == {3, 5} and set(subj_out.tolist()) == {101, 102}


def test_make_windows_rejects_unknown_normalize():
    values = np.zeros((504, 9), dtype=np.float32)
    labels = np.zeros(504, dtype=np.int64)
    with pytest.raises(ValueError, match="unknown normalize"):
        pamap2.make_windows(values, labels, labels, window=168, normalize="bogus")


def test_split_subjects_disjoint_deterministic_and_config():
    subj = np.repeat(np.arange(101, 110), 5)  # 9 subjects
    pool_a, test_a = pamap2._split_subjects(subj)
    pool_b, test_b = pamap2._split_subjects(subj)
    assert (pool_a, test_a) == (pool_b, test_b)         # deterministic
    assert len(test_a) == pamap2.N_TEST_SUBJECTS == 3
    assert len(pool_a) == 6                             # pool = remaining 6
    assert not (set(pool_a) & set(test_a))             # disjoint
    assert set(pool_a) | set(test_a) == set(range(101, 110))
    # a different seed yields a different held-out set (config flows in)
    _, test_seed1 = pamap2._split_subjects(subj, split_seed=1)
    assert set(test_a) != set(test_seed1)


def test_split_subjects_raises_when_pool_empty():
    subj = np.repeat(np.array([101, 102, 103]), 5)
    with pytest.raises(ValueError, match="no pool subjects"):
        pamap2._split_subjects(subj, n_test_subjects=3)


def test_per_subject_activity_support_counts_and_absences():
    y = np.array([0, 0, 3, 11], dtype=np.int64)
    subj = np.array([101, 101, 101, 102], dtype=np.int64)
    support = pamap2.per_subject_activity_support(y, subj)
    assert support[101]["lying"] == 2          # label 0
    assert support[101]["walking"] == 1        # label 3
    assert support[101]["rope_jumping"] == 0   # absent for 101
    assert support[102]["rope_jumping"] == 1   # label 11 present for 102


def _tiny_protocol(tmp_path, subjects=(101, 102, 103, 104), per_act=520):
    """Populate a Protocol/ dir with all 12 protocol activities per subject."""
    protocol = tmp_path / "Protocol"
    protocol.mkdir()
    for s in subjects:
        rows = np.concatenate([_make_rows(a, per_act, seed=s + a) for a in pamap2.PROTOCOL_ACTIVITY_IDS])
        (protocol / f"subject{s}.dat").write_text(_to_text(rows))
    return protocol


def test_record_metadata_records_config_and_support(tmp_path):
    protocol = _tiny_protocol(tmp_path)
    values, labels, subjects = pamap2.build_arrays(protocol)
    X, y, subj = pamap2.make_windows(values, labels, subjects, window=168, downsample_factor=3)
    pool_s, test_s = pamap2._split_subjects(subj, n_test_subjects=1)
    meta = pamap2._record_metadata(
        tmp_path, X, y, subj, pool_s, test_s,
        window=168, split_seed=0, n_test_subjects=1, normalize="per_window_z",
        downsample_hz=33, channels=pamap2.DEFAULT_CHANNELS,
    )
    saved = json.loads((tmp_path / "pamap2.json").read_text())
    assert saved["window_length"] == 168
    assert saved["downsample_hz"] == 33 and saved["downsample_factor"] == 3
    assert saved["n_classes"] == 12
    assert saved["channels"] == pamap2.DEFAULT_CHANNELS
    assert saved["license"] == "CC BY 4.0" and saved["doi"] == "10.24432/C5NW2H"
    assert saved["pool_windows_checksum"] and saved["test_windows_checksum"]
    assert "per_subject_activity_support" in saved
    assert meta["n_pool_windows"] > 0


def test_record_metadata_detects_drift(tmp_path):
    protocol = _tiny_protocol(tmp_path)
    values, labels, subjects = pamap2.build_arrays(protocol)

    def record(window):
        X, y, subj = pamap2.make_windows(values, labels, subjects, window=window, downsample_factor=3)
        pool_s, test_s = pamap2._split_subjects(subj, n_test_subjects=1)
        pamap2._record_metadata(
            tmp_path, X, y, subj, pool_s, test_s,
            window=window, split_seed=0, n_test_subjects=1, normalize="per_window_z",
            downsample_hz=33, channels=pamap2.DEFAULT_CHANNELS,
        )

    record(168)  # first write
    record(168)  # identical -> no error
    with pytest.raises(ValueError, match="metadata drift"):
        record(100)  # different window set -> drift caught


def test_load_pamap2_end_to_end(tmp_path, monkeypatch):
    protocol = _tiny_protocol(tmp_path, subjects=(101, 102, 103, 104))
    monkeypatch.setattr(pamap2, "_ensure_extracted", lambda *a, **k: protocol)
    pool, test = pamap2.load_pamap2(metadata_dir=tmp_path)
    # K=3 test / pool = 1 by default N_TEST_SUBJECTS=3 with 4 subjects
    assert set(pool.unique_subjects().tolist()).isdisjoint(test.unique_subjects().tolist())
    assert pool.n_subjects() == 1 and test.n_subjects() == 3
    assert pool.X.shape[1] == 9  # 9 accel channels
    # metadata written with the drift schema
    meta = json.loads((tmp_path / "pamap2.json").read_text())
    assert meta["dataset"] == "PAMAP2" and meta["n_classes"] == 12


def test_dispatch_registry_includes_pamap2():
    assert set(SUBJECT_LOADERS) >= {"UCI_HAR", "WISDM", "PAMAP2"}
    with pytest.raises(ValueError, match="unknown subject dataset"):
        load_subject_dataset("NOT_A_DATASET")
