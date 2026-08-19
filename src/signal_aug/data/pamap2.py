"""PAMAP2 Physical Activity Monitoring raw-signal loader for the subject-count
reduction study (issue #21 DS-2). Built to be the WISDM/UCI-HAR analogue: the
low-subject-count (~7-9 subjects) end-point of the cross-dataset "reduction vs
population size" comparison (artifacts/ds2_design.md).

PAMAP2 (Reiss & Stricker, 2012) is 3x Colibri IMU + heart-rate data: 9 subjects
(101-109), 100 Hz IMUs on hand/chest/ankle, 18 activities of which 12 form the
standard *protocol* subset. Raw files are one space-separated ``.dat`` per
subject with 54 columns.

Column layout (0-indexed), 54 columns total:
  * col 0        : timestamp (s)
  * col 1        : activityID (0 = transient, excluded)
  * col 2        : heart rate (bpm, ~9 Hz -> mostly NaN; excluded)
  * cols 3..19   : IMU hand   (17 cols)
  * cols 20..36  : IMU chest  (17 cols)
  * cols 37..53  : IMU ankle  (17 cols)
Each 17-col IMU block: temperature(1), accel +-16g(3), accel +-6g(3),
gyroscope(3), magnetometer(3), orientation quaternion(4, officially invalid).

Preprocessing judgment calls (recorded in artifacts/judgment_calls.yaml as
J-PAMAP2-*, and in data/metadata/pamap2.json):
  * channels = accelerometer only, +-16g, 9ch (hand/chest/ankle x/y/z). The +-6g
    duplicate range, gyroscope, magnetometer, HR and the invalid orientation
    quaternion are all excluded. Rationale: acceleration-domain parity with
    WISDM (3ch accel) / UCI HAR (accel+gyro), and +-16g avoids the +-6g range's
    saturation clipping (design 1.3/1.4). Config-selectable.
  * transient (activityID 0) and the 6 optional activities are dropped; only the
    12 protocol activities are windowed.
  * downsample 100 -> 33.3 Hz (stride-3 decimation) so the window's real-time
    length is comparable to WISDM/UCI HAR (a cross-dataset confound otherwise).
  * window = 168 samples (~5.0 s at 33.3 Hz), non-overlapping, within each
    contiguous (subject, activity) run -> single-label windows.
  * any window containing NaN in a selected channel (wireless dropout) is
    dropped (leak-free, no imputation).
  * per-window, per-channel z-normalisation (leak-free: each window uses only
    its own statistics), matching the WISDM loader.
  * fixed test subjects chosen by a seed-0 shuffle of the sorted subject ids
    (K=3 held-out), pool = the rest. ``exclude_subjects`` removes subjects whose
    protocol-activity support is deficient (a data-driven, pre-registerable call
    made from the measured per-subject support, NOT from accuracy).

License: CC BY 4.0. Source: Reiss & Stricker (2012), ISWC/PETRA; UCI ML
Repository dataset 231 (DOI:10.24432/C5NW2H).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/231/"
    "pamap2+physical+activity+monitoring.zip"
)

# --- Activity mapping: the 12 protocol activities -> contiguous labels 0..11 ---
# (design 1.1). transient (0) and the 6 optional activities are not listed here,
# so they are dropped rather than mapped.
PROTOCOL_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
PROTOCOL_ACTIVITY_NAMES = [
    "lying", "sitting", "standing", "walking", "running", "cycling",
    "nordic_walking", "ascending_stairs", "descending_stairs",
    "vacuum_cleaning", "ironing", "rope_jumping",
]
ACTIVITY_ID_TO_LABEL = {aid: i for i, aid in enumerate(PROTOCOL_ACTIVITY_IDS)}
ACTIVITIES = PROTOCOL_ACTIVITY_NAMES

# --- Channel column resolution ------------------------------------------------
# Per-IMU 0-based offset of each sensor axis within its 17-column block.
_IMU_BASE = {"hand": 3, "chest": 20, "ankle": 37}
_IMU_OFFSET = {
    "temp": 0,
    "acc16_x": 1, "acc16_y": 2, "acc16_z": 3,
    "acc6_x": 4, "acc6_y": 5, "acc6_z": 6,
    "gyro_x": 7, "gyro_y": 8, "gyro_z": 9,
    "mag_x": 10, "mag_y": 11, "mag_z": 12,
    "orient_0": 13, "orient_1": 14, "orient_2": 15, "orient_3": 16,
}
N_RAW_COLUMNS = 54

# Default channel set: +-16g accelerometer, 9 channels (design 1.4 option A).
DEFAULT_CHANNELS = [
    "hand_acc16_x", "hand_acc16_y", "hand_acc16_z",
    "chest_acc16_x", "chest_acc16_y", "chest_acc16_z",
    "ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z",
]

# --- Config-driven preprocessing defaults ------------------------------------
# The experiment config (config/experiments/pamap2_*.yaml -> dataset_params) is
# the source of truth; these constants only preserve behaviour when a caller
# omits a value, and are NOT where an experiment's conditions are set
# (CLAUDE.md: config-driven, no embedded experiment conditions).
NATIVE_HZ = 100.0
DOWNSAMPLE_HZ = 33          # -> stride-3 decimation, 33.3 Hz effective
WINDOW = 168                # ~5.0 s at 33.3 Hz, non-overlapping
N_TEST_SUBJECTS = 3         # fixed held-out subjects (K=3); pool = the rest
SPLIT_SEED = 0              # seed for the deterministic, pre-registered split
NORMALIZE = "per_window_z"  # per-window per-channel z-norm (leak-free); or "none"

LICENSE = "CC BY 4.0"
DOI = "10.24432/C5NW2H"
SOURCE = (
    "Reiss & Stricker (2012), ISWC/PETRA; UCI ML Repository dataset 231, "
    "DOI:10.24432/C5NW2H"
)


def resolve_channel_columns(channels: list[str]) -> list[int]:
    """Map channel names (e.g. 'hand_acc16_x') to 0-based raw-file columns.

    Raises on an unknown name or a name resolving to the invalid orientation /
    HR columns being requested by mistake (any name not in the registry).
    """
    cols = []
    for ch in channels:
        loc, _, axis = ch.partition("_")
        if loc not in _IMU_BASE or axis not in _IMU_OFFSET:
            raise ValueError(
                f"unknown PAMAP2 channel {ch!r}; expected '<hand|chest|ankle>_<sensor>' "
                f"with sensor in {sorted(_IMU_OFFSET)}"
            )
        cols.append(_IMU_BASE[loc] + _IMU_OFFSET[axis])
    return cols


def downsample_factor(downsample_hz: float) -> int:
    """Integer decimation factor from the native 100 Hz to ``downsample_hz``.

    ``downsample_hz=33`` (or 34) -> 3; the effective rate is NATIVE_HZ/factor
    (33.3 Hz), not exactly the requested value (design 1.4).
    """
    factor = int(round(NATIVE_HZ / float(downsample_hz)))
    if factor < 1:
        raise ValueError(f"downsample_hz={downsample_hz} exceeds native {NATIVE_HZ} Hz")
    return factor


# --- Raw parsing --------------------------------------------------------------
def parse_dat(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one subject's space-separated ``.dat`` text.

    Returns (data, activity) where data is (n, 54) float32 with NaN preserved
    (missing HR / wireless dropouts) and activity is (n,) int64 = column 1.
    Row order is preserved so contiguous (subject, activity) runs survive.
    """
    flat = np.fromstring(text, sep=" ", dtype=np.float64)
    n = flat.size // N_RAW_COLUMNS
    if n == 0:
        raise ValueError("no PAMAP2 rows parsed (empty or malformed .dat text)")
    data = flat[: n * N_RAW_COLUMNS].reshape(n, N_RAW_COLUMNS)
    activity = data[:, 1].astype(np.int64)
    return data.astype(np.float32), activity


def build_arrays(
    protocol_dir: str | Path,
    channels: list[str] | None = None,
    exclude_subjects: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read subject10X.dat files -> (values, labels, subjects), protocol rows only.

    values is (n, C) float32 (selected channels, NaN preserved), labels are
    contiguous 0..11, subjects are the integer subject id. Subjects are read in
    sorted id order and rows keep their in-file order, so make_windows can form
    contiguous (subject, activity) blocks. ``exclude_subjects`` drops whole
    subjects before any windowing.
    """
    channels = channels or DEFAULT_CHANNELS
    exclude = set(exclude_subjects or set())
    cols = resolve_channel_columns(channels)

    dats = sorted(Path(protocol_dir).glob("subject1*.dat"))
    if not dats:
        raise FileNotFoundError(f"no subject1*.dat files under {protocol_dir}")

    values, labels, subjects = [], [], []
    for dat in dats:
        subject = int(dat.stem.replace("subject", ""))
        if subject in exclude:
            continue
        data, activity = parse_dat(dat.read_text(encoding="utf-8", errors="ignore"))
        keep = np.array([a in ACTIVITY_ID_TO_LABEL for a in activity])
        if not keep.any():
            continue
        sel = data[keep][:, cols]
        lab = np.array([ACTIVITY_ID_TO_LABEL[a] for a in activity[keep]], dtype=np.int64)
        values.append(sel)
        labels.append(lab)
        subjects.append(np.full(len(lab), subject, dtype=np.int64))
    if not values:
        raise ValueError("no PAMAP2 protocol rows found across subjects")
    return (
        np.concatenate(values).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(subjects),
    )


def _znorm_window(win: np.ndarray) -> np.ndarray:
    """Per-channel z-normalise one (C, window) window (leak-free)."""
    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True)
    return ((win - mean) / (std + 1e-6)).astype(np.float32)


_NORMALIZERS = {
    "per_window_z": _znorm_window,
    "none": lambda win: win.astype(np.float32),
}


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    window: int = WINDOW,
    downsample_factor: int = 3,
    normalize: str = NORMALIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate then non-overlapping-window within each contiguous
    (subject, activity) run.

    ``values`` is (n, C). For each block the rows are decimated by
    ``downsample_factor`` (stride), then cut into non-overlapping windows of
    ``window`` decimated samples. A window with any NaN in a selected channel is
    dropped. Returns (X, y, subj) with X shape (m, C, window). ``normalize``
    selects the leak-free per-window normaliser ("per_window_z" or "none").
    """
    if normalize not in _NORMALIZERS:
        raise ValueError(
            f"unknown normalize mode {normalize!r}; known: {sorted(_NORMALIZERS)}"
        )
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
    norm = _NORMALIZERS[normalize]
    X, y, subj = [], [], []
    n = len(labels)
    start = 0
    while start < n:
        end = start
        while end < n and subjects[end] == subjects[start] and labels[end] == labels[start]:
            end += 1
        block = values[start:end][::downsample_factor]  # (m', C)
        n_win = len(block) // window
        for w in range(n_win):
            seg = block[w * window : (w + 1) * window].T  # (C, window)
            if not np.isfinite(seg).all():
                continue  # drop windows with NaN (wireless dropout)
            X.append(norm(seg))
            y.append(int(labels[start]))
            subj.append(int(subjects[start]))
        start = end
    if not X:
        raise ValueError("no PAMAP2 windows produced (window too large / all NaN?)")
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
    """Deterministic pre-registered test/pool subject split (WISDM-identical).

    Operates on whatever subjects survive windowing/exclusion: sort the unique
    ids, seed-``split_seed`` shuffle, first ``n_test_subjects`` -> test, rest ->
    pool. So excluding a subject shrinks the pool, not the test size.
    """
    unique = np.sort(np.unique(all_subjects))
    if n_test_subjects >= len(unique):
        raise ValueError(
            f"n_test_subjects={n_test_subjects} leaves no pool subjects "
            f"(only {len(unique)} subjects available)"
        )
    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(unique)
    test = sorted(int(s) for s in shuffled[:n_test_subjects])
    pool = sorted(int(s) for s in shuffled[n_test_subjects:])
    return pool, test


def per_subject_activity_support(
    labels: np.ndarray, subjects: np.ndarray
) -> dict[int, dict[str, int]]:
    """Window count per (subject, protocol-activity), for the subject-109 call.

    Records how many windows each subject contributes to each of the 12 protocol
    activities (0 = the activity is absent for that subject). Written to
    data/metadata/pamap2.json so the exclude/keep decision is data-driven and
    reproducible (design 5.2)."""
    support: dict[int, dict[str, int]] = {}
    for s in np.unique(subjects):
        mask = subjects == s
        counts = np.bincount(labels[mask], minlength=len(ACTIVITIES))
        support[int(s)] = {ACTIVITIES[i]: int(counts[i]) for i in range(len(ACTIVITIES))}
    return support


# --- Download / extraction ----------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_protocol(outer_zip: Path, protocol_dir: Path) -> None:
    """Extract Protocol/subject10X.dat from the (possibly nested) archive.

    UCI ships the outer static zip containing an inner ``PAMAP2_Dataset.zip``;
    both layouts are handled. Only the Protocol subject files are extracted
    (Optional/ activities are outside the 12-protocol subset)."""
    protocol_dir.mkdir(parents=True, exist_ok=True)

    def _extract_from(zf: zipfile.ZipFile) -> int:
        n = 0
        for name in zf.namelist():
            base = name.replace("\\", "/").split("/")[-1]
            if "Protocol" in name.replace("\\", "/") and base.startswith("subject") and base.endswith(".dat"):
                (protocol_dir / base).write_bytes(zf.read(name))
                n += 1
        return n

    with zipfile.ZipFile(outer_zip) as zf:
        extracted = _extract_from(zf)
        if extracted == 0:
            # nested zip layout: recurse one level into inner *.zip members
            for name in zf.namelist():
                if name.lower().endswith(".zip"):
                    with zipfile.ZipFile(BytesIO(zf.read(name))) as inner:
                        extracted += _extract_from(inner)
    if extracted == 0:
        raise FileNotFoundError(
            f"no Protocol/subject*.dat found in {outer_zip} (unexpected archive layout)"
        )


def _ensure_extracted(
    data_dir: str | Path, archive_sha256: str | None = None
) -> Path:
    """Download + extract PAMAP2 on first use; return the Protocol directory.

    The raw archive is large (~1.6 GB) and gitignored (data/raw/*). Subsequent
    loads reuse the extracted subject files."""
    root = Path(data_dir) / "PAMAP2"
    protocol_dir = root / "Protocol"
    if sorted(protocol_dir.glob("subject1*.dat")):
        return protocol_dir
    root.mkdir(parents=True, exist_ok=True)
    outer = root / "pamap2.zip"
    if not (outer.exists() and outer.stat().st_size > 1_000_000):
        urllib.request.urlretrieve(DATA_URL, outer)
    if archive_sha256:
        actual = _sha256(outer)
        if actual != archive_sha256:
            raise ValueError(
                f"PAMAP2 archive checksum mismatch for {outer}: "
                f"expected {archive_sha256}, got {actual}"
            )
    _extract_protocol(outer, protocol_dir)
    return protocol_dir


def load_pamap2(
    data_dir: str | Path = "data/raw",
    metadata_dir: str | Path = "data/metadata",
    *,
    window: int = WINDOW,
    split_seed: int = SPLIT_SEED,
    n_test_subjects: int = N_TEST_SUBJECTS,
    normalize: str = NORMALIZE,
    downsample_hz: float = DOWNSAMPLE_HZ,
    channels: list[str] | None = None,
    exclude_subjects: list[int] | None = None,
    archive_sha256: str | None = None,
):
    """Return (pool, test) as SubjectSplits.

    pool = training subjects used to build subject-count curves.
    test = fixed held-out subjects (K = ``n_test_subjects``), subject-disjoint.

    The preprocessing knobs (``window``, ``split_seed``, ``n_test_subjects``,
    ``normalize``, ``downsample_hz``, ``channels``, ``exclude_subjects``) are
    config-driven (dataset_params); the defaults reproduce the pre-registered
    setup (168-sample windows @33.3 Hz, seed-0 K=3 split, 9-ch +-16g accel,
    per-window z-norm). ``archive_sha256`` optionally pins the downloaded
    archive's hash.
    """
    from signal_aug.data.subject import SubjectSplits

    channels = channels or DEFAULT_CHANNELS
    protocol_dir = _ensure_extracted(data_dir, archive_sha256=archive_sha256)

    values, labels, subjects = build_arrays(protocol_dir, channels, set(exclude_subjects or []))
    factor = downsample_factor(downsample_hz)
    X, y, subj = make_windows(
        values, labels, subjects,
        window=window, downsample_factor=factor, normalize=normalize,
    )

    pool_subjects, test_subjects = _split_subjects(
        subj, n_test_subjects=n_test_subjects, split_seed=split_seed
    )
    overlap = set(pool_subjects) & set(test_subjects)
    if overlap:
        raise ValueError(f"PAMAP2 pool/test subject overlap: {overlap}")

    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)

    _record_metadata(
        Path(metadata_dir), X, y, subj, pool_subjects, test_subjects,
        window=window, split_seed=split_seed, n_test_subjects=n_test_subjects,
        normalize=normalize, downsample_hz=downsample_hz, channels=channels,
    )

    pool = SubjectSplits("PAMAP2_pool", X[pool_mask], y[pool_mask], subj[pool_mask])
    test = SubjectSplits("PAMAP2_test", X[test_mask], y[test_mask], subj[test_mask])
    return pool, test


# Config-relevant fields whose change means the recorded window set is stale.
_METADATA_DRIFT_FIELDS = (
    "window_length", "split_seed", "n_test_subjects", "normalize",
    "downsample_hz", "channels",
    "n_pool_windows", "n_test_windows",
    "pool_windows_checksum", "test_windows_checksum",
)


def _record_metadata(
    metadata_dir: Path, X, y, subj, pool_subjects, test_subjects,
    *, window: int, split_seed: int, n_test_subjects: int, normalize: str,
    downsample_hz: float, channels: list[str],
) -> dict:
    """Write data/metadata/pamap2.json on first build; afterwards verify.

    Records the config knobs, per-subject protocol-activity support, and a
    content hash of the actual pool/test window arrays. On a subsequent load the
    freshly built metadata is compared with the recorded file: identical -> left
    in place (no silent overwrite), drifted -> raise (spec section 7)."""
    from signal_aug.data.subject import windows_checksum

    metadata_dir.mkdir(parents=True, exist_ok=True)
    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)
    factor = downsample_factor(downsample_hz)
    channel_columns = resolve_channel_columns(channels)
    meta = {
        "dataset": "PAMAP2",
        "license": LICENSE,
        "doi": DOI,
        "source": SOURCE,
        "url": DATA_URL,
        "channels": list(channels),
        "channel_columns": channel_columns,
        "native_hz": NATIVE_HZ,
        "downsample_hz": int(downsample_hz),
        "downsample_factor": int(factor),
        "effective_hz": round(NATIVE_HZ / factor, 3),
        "window_length": int(window),
        "n_classes": len(ACTIVITIES),
        "activities": ACTIVITIES,
        "protocol_activity_ids": PROTOCOL_ACTIVITY_IDS,
        "preprocessing": (
            "protocol-12 activities only; transient/optional dropped; "
            f"stride-{factor} decimation; non-overlapping {window}-sample windows "
            f"within (subject,activity) runs; NaN windows dropped; normalize={normalize}"
        ),
        "normalize": normalize,
        "split_seed": int(split_seed),
        "n_test_subjects": int(n_test_subjects),
        "pool_subjects": [int(s) for s in pool_subjects],
        "test_subjects": [int(s) for s in test_subjects],
        "n_pool_windows": int(pool_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "pool_windows_checksum": windows_checksum(X[pool_mask]),
        "test_windows_checksum": windows_checksum(X[test_mask]),
        "per_subject_activity_support": per_subject_activity_support(y, subj),
    }
    path = metadata_dir / "pamap2.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing and "pool_windows_checksum" in existing:
            drift = {
                k: {"recorded": existing.get(k), "current": meta[k]}
                for k in _METADATA_DRIFT_FIELDS
                if existing.get(k) != meta[k]
            }
            if drift:
                raise ValueError(
                    f"PAMAP2 dataset metadata drift vs {path}: {drift}. The recorded "
                    "window set no longer matches the freshly built one. If this "
                    "reconfiguration is intended, delete the stale metadata file to "
                    "regenerate it."
                )
            return meta  # matches the first-written record; leave it untouched
    path.write_text(json.dumps(meta, indent=2))
    return meta
