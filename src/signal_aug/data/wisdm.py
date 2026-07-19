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

import hashlib
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

# Defaults for the config-driven preprocessing knobs. The experiment config
# (config/experiments/wisdm_*.yaml -> dataset_params) is the source of truth;
# these constants only preserve backward-compatible behaviour when a caller does
# not supply a value, and are NOT the place to change an experiment's conditions
# (CLAUDE.md: config-driven, no embedded experiment conditions).
WINDOW = 200          # ~10 s at 20 Hz, matches the WISDM-standard window
N_TEST_SUBJECTS = 12  # fixed held-out subjects (of 36); pool = remaining 24
SPLIT_SEED = 0        # seed for the deterministic, pre-registered test split
NORMALIZE = "per_window_z"  # per-window per-channel z-norm (leak-free); or "none"

LICENSE = "CC BY 4.0"
SOURCE = "Kwapisz, Weiss & Moore (2011), SIGKDD Explorations 12(2); WISDM v1.1"


def _download(dest: Path, expected_sha256: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 1_000_000):
        urllib.request.urlretrieve(DATA_URL, dest)
    if expected_sha256:
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"WISDM tar checksum mismatch for {dest}: "
                f"expected {expected_sha256}, got {actual}"
            )


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


# Registry of leak-free per-window normalisers selectable from the config
# (dataset_params.normalize). Each takes/returns a (3, window) float32 array.
_NORMALIZERS = {
    "per_window_z": _znorm_window,
    "none": lambda win: win.astype(np.float32),
}


def make_windows(
    xyz: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    window: int = WINDOW,
    normalize: str = NORMALIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-overlapping windows within each contiguous (subject, activity) run.

    Returns (X, y, subj) with X shape (n, 3, window). ``normalize`` selects the
    per-window normalisation ("per_window_z" default, or "none"); both are
    leak-free because each window uses only its own statistics.
    """
    if normalize not in _NORMALIZERS:
        raise ValueError(
            f"unknown normalize mode {normalize!r}; known: {sorted(_NORMALIZERS)}"
        )
    norm = _NORMALIZERS[normalize]
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
            X.append(norm(seg))
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


def _split_subjects(
    all_subjects: np.ndarray,
    n_test_subjects: int = N_TEST_SUBJECTS,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Deterministic pre-registered test/pool subject split.

    The number of held-out test subjects and the shuffle seed come from the
    config (dataset_params); the defaults reproduce the pre-registered seed-0,
    12-subject split.
    """
    unique = np.sort(np.unique(all_subjects))
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(unique)
    test = sorted(int(s) for s in shuffled[:n_test_subjects])
    pool = sorted(int(s) for s in shuffled[n_test_subjects:])
    return pool, test


def load_wisdm(
    data_dir: str | Path = "data/raw",
    metadata_dir: str | Path = "data/metadata",
    *,
    window: int = WINDOW,
    split_seed: int = SPLIT_SEED,
    n_test_subjects: int = N_TEST_SUBJECTS,
    normalize: str = NORMALIZE,
    tar_sha256: str | None = None,
):
    """Return (pool, test) as SubjectSplits.

    pool = training subjects used to build subject-count curves.
    test = fixed held-out subjects, subject-disjoint from pool.

    The preprocessing knobs (``window``, ``split_seed``, ``n_test_subjects``,
    ``normalize``) are config-driven (dataset_params); the defaults reproduce the
    pre-registered setup (200-sample windows, seed-0 12-subject split, per-window
    z-norm). ``tar_sha256`` optionally pins the downloaded archive's hash.
    """
    from signal_aug.data.subject import SubjectSplits

    data_dir = Path(data_dir)
    tar_path = data_dir / "WISDM" / "wisdm_ar_v1.1.tar.gz"
    _download(tar_path, expected_sha256=tar_sha256)
    text = _read_raw_text(tar_path)

    xyz, labels, subjects = parse_raw(text)
    X, y, subj = make_windows(xyz, labels, subjects, window=window, normalize=normalize)

    pool_subjects, test_subjects = _split_subjects(
        subj, n_test_subjects=n_test_subjects, split_seed=split_seed
    )
    overlap = set(pool_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"WISDM pool/test subject overlap: {overlap}")

    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)

    _record_metadata(
        Path(metadata_dir), X, y, subj, pool_subjects, test_subjects,
        window=window, split_seed=split_seed, n_test_subjects=n_test_subjects,
        normalize=normalize,
    )

    pool = SubjectSplits("WISDM_pool", X[pool_mask], y[pool_mask], subj[pool_mask])
    test = SubjectSplits("WISDM_test", X[test_mask], y[test_mask], subj[test_mask])
    return pool, test


# Config-relevant fields whose change means the recorded window set is stale.
_METADATA_DRIFT_FIELDS = (
    "window_length", "split_seed", "n_test_subjects", "normalize",
    "n_pool_windows", "n_test_windows",
    "pool_windows_checksum", "test_windows_checksum",
)


def _record_metadata(
    metadata_dir: Path, X, y, subj, pool_subjects, test_subjects,
    *, window: int, split_seed: int, n_test_subjects: int, normalize: str,
) -> dict:
    """Write data/metadata/wisdm.json on first build; afterwards verify.

    Records the config knobs and a content hash of the actual pool/test window
    arrays. On a subsequent load the freshly built metadata is compared against
    the recorded file: identical -> left in place (no silent overwrite), drifted
    -> raise, so a changed windowing cannot go unnoticed (spec section 7).
    """
    from signal_aug.data.subject import windows_checksum

    metadata_dir.mkdir(parents=True, exist_ok=True)
    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)
    meta = {
        "dataset": "WISDM",
        "version": "v1.1",
        "license": LICENSE,
        "source": SOURCE,
        "url": DATA_URL,
        "channels": CHANNELS,
        "window_length": int(window),
        "sampling_hz": 20,
        "n_classes": len(ACTIVITIES),
        "activities": ACTIVITIES,
        "preprocessing": (
            "non-overlapping windows within (subject,activity) runs; "
            f"normalize={normalize}"
        ),
        "normalize": normalize,
        "split_seed": int(split_seed),
        "n_test_subjects": int(n_test_subjects),
        "pool_subjects": pool_subjects,
        "test_subjects": test_subjects,
        "n_pool_windows": int(pool_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "pool_windows_checksum": windows_checksum(X[pool_mask]),
        "test_windows_checksum": windows_checksum(X[test_mask]),
    }
    path = metadata_dir / "wisdm.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        # Only enforce against records already carrying the checksum schema; a
        # pre-checksum file is migrated by overwrite (backward compatible).
        if existing and "pool_windows_checksum" in existing:
            drift = {
                k: {"recorded": existing.get(k), "current": meta[k]}
                for k in _METADATA_DRIFT_FIELDS
                if existing.get(k) != meta[k]
            }
            if drift:
                raise ValueError(
                    f"WISDM dataset metadata drift vs {path}: {drift}. The recorded "
                    "window set no longer matches the freshly built one. If this "
                    "reconfiguration is intended, delete the stale metadata file to "
                    "regenerate it."
                )
            return meta  # matches the first-written record; leave it untouched
    path.write_text(json.dumps(meta, indent=2))
    return meta
