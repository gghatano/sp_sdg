"""WISDM v1.1 (Actitracker) raw-signal loader for the subject-count reduction
study (Phase 4-5 / issue #21, DS-1).

WISDM v1.1 (Kwapisz et al., 2011) is phone-accelerometer data: 36 subjects,
3 channels (x/y/z), 6 activities (Walking, Jogging, Upstairs, Downstairs,
Sitting, Standing), sampled at ~20 Hz. It is the closest analogue to UCI HAR
(6 activities, accelerometry) and serves as the second subject-labelled dataset
for testing whether the subject-count reduction result generalises beyond a
single dataset (issue #7 candidate 3, issue #21 DS-1).

There is no official train/test split, so we hold out a fixed, subject-disjoint
set of test subjects (spec section 8: subjects never straddle train and test)
and build subject-count learning curves from the remaining pool subjects.

Preprocessing judgment calls (recorded in artifacts/judgment_calls.yaml):
  * window = 200 samples (~10 s at 20 Hz), non-overlapping, within each
    contiguous (subject, activity) block -> single-label windows, no
    within-set window overlap.
  * per-window, per-channel z-normalisation (leak-free: each window uses only
    its own statistics), because raw WISDM acceleration (~m/s^2) is on a much
    larger scale than the CNN's first conv expects.
  * fixed test subjects chosen by a seed-0 shuffle of the sorted subject ids
    (12 of 36 ~= the UCI HAR 30% test ratio), pool = remaining 24.

License: CC BY 4.0. Source: Kwapisz, Weiss & Moore (2011), SIGKDD Explorations.
Malformed rows (~1% of the raw file) are skipped; see rWISDM (arXiv:2305.10222)
for a discussion of the raw-file issues.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
RAW_MEMBER = "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
ACTIVITY_TO_LABEL = {a: i for i, a in enumerate(ACTIVITIES)}
CHANNELS = ["accel_x", "accel_y", "accel_z"]

WINDOW = 200          # ~10 s at 20 Hz, matches the WISDM-standard window
N_TEST_SUBJECTS = 12  # fixed held-out subjects (of 36); pool = remaining 24
SPLIT_SEED = 0        # seed for the deterministic, pre-registered test split

LICENSE = "CC BY 4.0"
SOURCE = "Kwapisz, Weiss & Moore (2011), SIGKDD Explorations 12(2); WISDM v1.1"


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    urllib.request.urlretrieve(DATA_URL, dest)


def _read_raw_text(tar_path: Path) -> str:
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.getmember(RAW_MEMBER)
        fobj = tf.extractfile(member)
        if fobj is None:
            raise FileNotFoundError(f"{RAW_MEMBER} not found in {tar_path}")
        return fobj.read().decode("utf-8", errors="ignore")


def parse_raw(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the semicolon-terminated raw file into per-sample arrays.

    Returns (xyz, labels, subjects) where xyz is (n, 3) float32, labels/subjects
    are (n,) int64. Malformed records are skipped. Order within the file is
    preserved so that contiguous (subject, activity) runs can be windowed.
    """
    xyz, labels, subjects = [], [], []
    for rec in text.replace("\n", "").split(";"):
        rec = rec.strip()
        if not rec:
            continue
        parts = rec.split(",")
        if len(parts) != 6:
            continue
        user, activity, _ts, x, y, z = parts
        if activity not in ACTIVITY_TO_LABEL:
            continue
        try:
            u = int(user)
            fx, fy, fz = float(x), float(y), float(z)
        except ValueError:
            continue
        xyz.append((fx, fy, fz))
        labels.append(ACTIVITY_TO_LABEL[activity])
        subjects.append(u)
    if not xyz:
        raise ValueError("no valid WISDM records parsed")
    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
    )


def _znorm_window(win: np.ndarray) -> np.ndarray:
    """Per-channel z-normalise one (3, WINDOW) window (leak-free)."""
    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True)
    return ((win - mean) / (std + 1e-6)).astype(np.float32)


def make_windows(
    xyz: np.ndarray, labels: np.ndarray, subjects: np.ndarray, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-overlapping windows within each contiguous (subject, activity) run.

    Returns (X, y, subj) with X shape (n, 3, window), z-normalised per window.
    """
    X, y, subj = [], [], []
    n = len(labels)
    start = 0
    while start < n:
        end = start
        while end < n and subjects[end] == subjects[start] and labels[end] == labels[start]:
            end += 1
        # window this contiguous block
        block = xyz[start:end]  # (m, 3)
        n_win = len(block) // window
        for w in range(n_win):
            seg = block[w * window : (w + 1) * window].T  # (3, window)
            X.append(_znorm_window(seg))
            y.append(int(labels[start]))
            subj.append(int(subjects[start]))
        start = end
    if not X:
        raise ValueError("no WISDM windows produced (window too large?)")
    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(subj, dtype=np.int64),
    )


def _split_subjects(all_subjects: np.ndarray) -> tuple[list[int], list[int]]:
    """Deterministic pre-registered test/pool subject split (seed-0)."""
    unique = np.sort(np.unique(all_subjects))
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(unique)
    test = sorted(int(s) for s in shuffled[:N_TEST_SUBJECTS])
    pool = sorted(int(s) for s in shuffled[N_TEST_SUBJECTS:])
    return pool, test


def load_wisdm(data_dir: str | Path = "data/raw", metadata_dir: str | Path = "data/metadata"):
    """Return (pool, test) as SubjectSplits.

    pool = 24 training subjects used to build subject-count curves.
    test = 12 fixed held-out subjects, subject-disjoint from pool.
    """
    from signal_aug.data.subject import SubjectSplits

    data_dir = Path(data_dir)
    tar_path = data_dir / "WISDM" / "wisdm_ar_v1.1.tar.gz"
    _download(tar_path)
    text = _read_raw_text(tar_path)

    xyz, labels, subjects = parse_raw(text)
    X, y, subj = make_windows(xyz, labels, subjects)

    pool_subjects, test_subjects = _split_subjects(subj)
    overlap = set(pool_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"WISDM pool/test subject overlap: {overlap}")

    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)

    _record_metadata(Path(metadata_dir), X, y, subj, pool_subjects, test_subjects)

    pool = SubjectSplits("WISDM_pool", X[pool_mask], y[pool_mask], subj[pool_mask])
    test = SubjectSplits("WISDM_test", X[test_mask], y[test_mask], subj[test_mask])
    return pool, test


def _record_metadata(metadata_dir: Path, X, y, subj, pool_subjects, test_subjects) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "WISDM",
        "version": "v1.1",
        "license": LICENSE,
        "source": SOURCE,
        "url": DATA_URL,
        "channels": CHANNELS,
        "window_length": int(WINDOW),
        "sampling_hz": 20,
        "n_classes": len(ACTIVITIES),
        "activities": ACTIVITIES,
        "preprocessing": "non-overlapping windows within (subject,activity) runs; per-window per-channel z-norm",
        "split_seed": SPLIT_SEED,
        "pool_subjects": pool_subjects,
        "test_subjects": test_subjects,
        "n_pool_windows": int(np.isin(subj, pool_subjects).sum()),
        "n_test_windows": int(np.isin(subj, test_subjects).sum()),
    }
    (metadata_dir / "wisdm.json").write_text(json.dumps(meta, indent=2))
