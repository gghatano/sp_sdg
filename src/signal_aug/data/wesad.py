"""WESAD (Wearable Stress and Affect Detection) raw-signal loader for the
subject-count reduction study (issue #21 DS-3). Built as the WISDM/PAMAP2/UCI-HAR
analogue on a *non-HAR physiological* signal type: instead of accelerometry /
activity recognition, WESAD is chest physiology (ECG/EDA/EMG/RESP/TEMP) and a
3-class affect task (baseline / stress / amusement). It tests whether the
"augmentation does not reduce the required subject count" result generalises
across signal *type*, not just across HAR datasets (artifacts/ds3_design.md).

WESAD (Schmidt et al., 2018) is 15 subjects wearing a chest RespiBAN (all
channels 700 Hz) and a wrist Empatica E4. This loader uses ONLY the 5 chest
physiological channels (ECG/EDA/EMG/RESP/TEMP); the chest 3-axis accelerometer
(motion = HAR-like) and the wrist E4 are excluded so the "non-HAR physiological"
claim is not diluted (design 2.2). The study protocol yields, per subject, one
contiguous run per affect state; the 700 Hz ``label`` array is synchronised to
the chest signals.

Label codes (readme): 0 = transient/undefined, 1 = baseline, 2 = stress,
3 = amusement, 4 = meditation, 5/6/7 = ignore. The 3-class task keeps ONLY
{1,2,3} and drops {0,4,5,6,7}; kept labels are remapped to contiguous 0..2.

Preprocessing judgment calls (recorded in artifacts/judgment_calls.yaml as
J-WESAD-*, and in data/metadata/wesad.json):
  * channels = chest physiology, 5ch (ECG/EDA/EMG/RESP/TEMP). Chest ACC and the
    wrist E4 are excluded. Config-selectable.
  * labels {1,2,3} only (baseline/stress/amusement); {0,4,5,6,7} dropped.
  * downsample 700 -> ~70 Hz via an anti-aliasing low-pass filter followed by
    decimation (scipy.signal.decimate, zero-phase IIR), NOT naive stride
    decimation: the raw 700 Hz would give ~2100-sample windows and simple
    striding aliases the ECG/EMG high-frequency content (design 1.4/2.2).
  * window = 2100 decimated samples (~30 s at 70 Hz), non-overlapping by
    default, within each contiguous (subject, label) run -> single-label
    windows, subjects never straddling train/test. ``overlap`` (fraction) is a
    config knob: if 30 s non-overlapping windows are too few to estimate N* it
    may be raised (e.g. 0.5) as a judgment call; subject-disjointness is always
    kept and the overlap is noted (design 2.2).
  * any window with a NaN in a selected channel is dropped (leak-free, no
    imputation). E4 is not used, so this is a safety guard.
  * per-window, per-channel z-normalisation (leak-free: each window uses only
    its own statistics), matching the WISDM/PAMAP2 loaders. Physiological
    channels differ by orders of magnitude in scale (ECG mV vs EDA uS vs
    TEMP degC), so this is required.
  * fixed test subjects chosen by a seed-0 shuffle of the sorted subject ids
    (K=5 held-out of 15), pool = the remaining 10, same procedure as
    WISDM/PAMAP2.

License: scientific / non-commercial use WITH ATTRIBUTION (this is NOT CC BY 4.0
like the HAR datasets, and redistribution is NOT granted) -> the raw .pkl data
is never committed; only checksums + metadata are recorded (design 1.5).
Source: Schmidt, Reiss, Duerichen, Marberger & Van Laerhoven (2018), ICMI 2018,
DOI:10.1145/3242969.3242985. The per-subject .pkl files are Python-2 pickles and
MUST be read with ``pickle.load(f, encoding="latin1")`` (a known pitfall).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.signal import decimate

# The author-distributed Sciebo (ownCloud) share of the ~2.5 GB zip. The UCI
# mirror #465's static zip is only a POINTER file (a WESAD.txt with this link),
# not the data, so we fetch the real archive directly. The share token has
# changed at least once historically; the sha256 (archive_sha256, config-pinned)
# is what actually reproduces the exact data, not the URL.
DATA_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
SOURCE_MIRROR = (
    "UCI ML Repository #465 (static zip is a pointer to the Sciebo link, not the data)"
)

# --- Task: the 3 affect states -> contiguous labels 0..2 (design 2.1) ---
# transient (0), meditation (4) and the ignore codes (5/6/7) are not listed here,
# so they are dropped rather than mapped.
LABEL_IDS = [1, 2, 3]
CLASS_NAMES = ["baseline", "stress", "amusement"]
LABEL_TO_CLASS = {lid: i for i, lid in enumerate(LABEL_IDS)}
N_CLASSES = len(LABEL_IDS)

# --- Channel resolution -------------------------------------------------------
# Channel name (config) -> key inside signal['chest'] of the .pkl. The chest
# physiological set; ACC (motion) is deliberately absent so it cannot be selected
# by the default and dilute the non-HAR claim.
_CHEST_KEY = {
    "ECG": "ECG",
    "EDA": "EDA",
    "EMG": "EMG",
    "RESP": "Resp",
    "TEMP": "Temp",
}
DEFAULT_CHANNELS = ["ECG", "EDA", "EMG", "RESP", "TEMP"]
MODALITY = "chest"

# --- Config-driven preprocessing defaults ------------------------------------
# The experiment config (config/experiments/wesad_*.yaml -> dataset_params) is
# the source of truth; these constants only preserve behaviour when a caller
# omits a value, and are NOT where an experiment's conditions are set
# (CLAUDE.md: config-driven, no embedded experiment conditions).
NATIVE_HZ = 700.0
DOWNSAMPLE_HZ = 70          # -> factor-10 decimation, 70 Hz effective
WINDOW = 2100               # ~30 s at 70 Hz, non-overlapping
OVERLAP = 0.0               # window overlap fraction (0 = non-overlapping)
N_TEST_SUBJECTS = 5         # fixed held-out subjects (K=5 of 15); pool = the rest
SPLIT_SEED = 0              # seed for the deterministic, pre-registered split
NORMALIZE = "per_window_z"  # per-window per-channel z-norm (leak-free); or "none"

LICENSE = (
    "Scientific / non-commercial use with attribution (NOT CC BY 4.0; "
    "redistribution not granted)"
)
DOI = "10.1145/3242969.3242985"
SOURCE = (
    "Schmidt, Reiss, Duerichen, Marberger & Van Laerhoven (2018), "
    "Introducing WESAD, ICMI 2018, pp. 400-408, DOI:10.1145/3242969.3242985; "
    "UCI ML Repository dataset 465"
)


def resolve_chest_channels(channels: list[str]) -> list[str]:
    """Map channel names (e.g. 'ECG', 'RESP') to signal['chest'] .pkl keys.

    Raises on an unknown name, in particular if 'ACC' or a wrist channel is
    requested (they are not in the chest-physiology registry by design)."""
    keys = []
    for ch in channels:
        key = _CHEST_KEY.get(ch.upper())
        if key is None:
            raise ValueError(
                f"unknown WESAD chest channel {ch!r}; expected one of "
                f"{sorted(_CHEST_KEY)} (chest physiology only; ACC/wrist excluded)"
            )
        keys.append(key)
    return keys


def downsample_factor(downsample_hz: float) -> int:
    """Integer decimation factor from native 700 Hz to ``downsample_hz``.

    ``downsample_hz=70`` -> 10; the effective rate is NATIVE_HZ/factor (70 Hz),
    which for a non-integer divisor is not exactly the requested value."""
    factor = int(round(NATIVE_HZ / float(downsample_hz)))
    if factor < 1:
        raise ValueError(f"downsample_hz={downsample_hz} exceeds native {NATIVE_HZ} Hz")
    return factor


# --- Raw parsing --------------------------------------------------------------
def parse_pkl(raw: dict, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract (values, labels) for the selected chest channels from one loaded
    WESAD subject dict.

    ``raw`` is the dict from ``pickle.load(f, encoding="latin1")``: it has
    ``signal['chest'][KEY]`` arrays of shape (n, 1) and a ``label`` array (n,)
    synchronised at 700 Hz. Returns (values, labels) where values is (n, C)
    float32 (selected channels, NaN preserved) and labels is (n,) int64 (raw
    label codes, not yet filtered/remapped). Row order is preserved so contiguous
    (subject, label) runs survive downstream."""
    keys = resolve_chest_channels(channels)
    chest = raw["signal"]["chest"]
    cols = []
    for key in keys:
        arr = np.asarray(chest[key], dtype=np.float32).reshape(len(raw["label"]), -1)
        if arr.shape[1] != 1:
            raise ValueError(
                f"WESAD chest channel {key!r} has {arr.shape[1]} columns, expected 1"
            )
        cols.append(arr)
    values = np.concatenate(cols, axis=1).astype(np.float32)  # (n, C)
    labels = np.asarray(raw["label"]).reshape(-1).astype(np.int64)
    if len(labels) != len(values):
        raise ValueError(
            f"WESAD label length {len(labels)} != signal length {len(values)}"
        )
    return values, labels


def _subject_id(stem: str) -> int:
    """'S2' / 'S17' -> 2 / 17. Raises on an unexpected filename stem."""
    if not (stem and stem[0].upper() == "S" and stem[1:].isdigit()):
        raise ValueError(f"unexpected WESAD subject file stem {stem!r} (expected 'S<int>')")
    return int(stem[1:])


def build_arrays(
    data_dir: str | Path,
    channels: list[str] | None = None,
    exclude_subjects: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read SX.pkl files -> (values, labels, subjects), keeping only labels
    {1,2,3} (baseline/stress/amusement) remapped to 0..2.

    values is (n, C) float32 (selected chest channels, NaN preserved), labels are
    contiguous 0..2, subjects are the integer subject id. Files are read in sorted
    id order and rows keep their in-file order, so make_windows forms contiguous
    (subject, label) blocks. ``exclude_subjects`` drops whole subjects before any
    windowing."""
    channels = channels or DEFAULT_CHANNELS
    exclude = set(exclude_subjects or set())

    pkls = sorted(
        Path(data_dir).glob("S*.pkl"), key=lambda p: _subject_id(p.stem)
    )
    if not pkls:
        raise FileNotFoundError(f"no S*.pkl files under {data_dir}")

    values, labels, subjects = [], [], []
    for pkl in pkls:
        subject = _subject_id(pkl.stem)
        if subject in exclude:
            continue
        with pkl.open("rb") as f:
            raw = pickle.load(f, encoding="latin1")
        vals, lab = parse_pkl(raw, channels)
        keep = np.isin(lab, LABEL_IDS)
        if not keep.any():
            continue
        sel = vals[keep]
        remapped = np.array([LABEL_TO_CLASS[int(a)] for a in lab[keep]], dtype=np.int64)
        values.append(sel)
        labels.append(remapped)
        subjects.append(np.full(len(remapped), subject, dtype=np.int64))
    if not values:
        raise ValueError("no WESAD {1,2,3}-label rows found across subjects")
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


def _decimate_block(block: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased downsample of one contiguous (m, C) block by ``factor``.

    Uses scipy.signal.decimate (zero-phase IIR low-pass then decimation) along
    the time axis, NOT naive striding, so the 700 Hz ECG/EMG content above the
    new Nyquist is filtered out instead of aliasing back into band. factor=1 is a
    no-op. Blocks shorter than the filter's padding cannot be decimated and are
    returned empty (they yield no windows)."""
    if factor <= 1:
        return block.astype(np.float32)
    # zero-phase IIR (Chebyshev-I order 8) filtfilt needs > ~3*order+1 samples.
    if len(block) <= 30:
        return np.empty((0, block.shape[1]), dtype=np.float32)
    deci = decimate(block, factor, axis=0, ftype="iir", zero_phase=True)
    return np.asarray(deci, dtype=np.float32)


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    window: int = WINDOW,
    downsample_factor: int = 10,
    normalize: str = NORMALIZE,
    overlap: float = OVERLAP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anti-alias-decimate then window within each contiguous (subject, label)
    run.

    ``values`` is (n, C). For each block the rows are decimated (anti-aliased) by
    ``downsample_factor``, then cut into windows of ``window`` decimated samples
    with step ``round(window * (1 - overlap))`` (overlap=0 -> non-overlapping). A
    window with any NaN in a selected channel is dropped. Returns (X, y, subj)
    with X shape (m, C, window). ``normalize`` selects the leak-free per-window
    normaliser ("per_window_z" or "none")."""
    if normalize not in _NORMALIZERS:
        raise ValueError(
            f"unknown normalize mode {normalize!r}; known: {sorted(_NORMALIZERS)}"
        )
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    norm = _NORMALIZERS[normalize]
    step = max(1, int(round(window * (1.0 - overlap))))
    X, y, subj = [], [], []
    n = len(labels)
    start = 0
    while start < n:
        end = start
        while end < n and subjects[end] == subjects[start] and labels[end] == labels[start]:
            end += 1
        block = _decimate_block(values[start:end], downsample_factor)  # (m', C)
        w_start = 0
        while w_start + window <= len(block):
            seg = block[w_start : w_start + window].T  # (C, window)
            w_start += step
            if not np.isfinite(seg).all():
                continue  # drop windows with NaN (safety guard)
            X.append(norm(seg))
            y.append(int(labels[start]))
            subj.append(int(subjects[start]))
        start = end
    if not X:
        raise ValueError("no WESAD windows produced (window too large / all NaN?)")
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
    """Deterministic pre-registered test/pool subject split (WISDM/PAMAP2-identical).

    Sort the unique ids, seed-``split_seed`` shuffle, first ``n_test_subjects`` ->
    test, rest -> pool. Excluding a subject shrinks the pool, not the test size."""
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


def per_subject_class_support(
    labels: np.ndarray, subjects: np.ndarray
) -> dict[int, dict[str, int]]:
    """Window count per (subject, affect-class), including the minority
    (amusement) support at small N. Written to data/metadata/wesad.json so the
    class balance is auditable and reproducible (design 1.4/2.3)."""
    support: dict[int, dict[str, int]] = {}
    for s in np.unique(subjects):
        mask = subjects == s
        counts = np.bincount(labels[mask], minlength=N_CLASSES)
        support[int(s)] = {CLASS_NAMES[i]: int(counts[i]) for i in range(N_CLASSES)}
    return support


# --- Download / extraction ----------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_pkls(outer_zip: Path, dest_dir: Path) -> int:
    """Extract SX.pkl subject files from the (possibly nested) WESAD archive.

    UCI ships an outer static zip that may contain the subject folders directly
    (``WESAD/S2/S2.pkl``) or a nested ``WESAD.zip``; both layouts are handled.
    Only per-subject ``S<int>.pkl`` files are extracted, flattened to
    ``dest_dir/S<int>.pkl``."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _is_subject_pkl(base: str) -> bool:
        if not (base.startswith("S") and base.endswith(".pkl")):
            return False
        return base[1:-4].isdigit()

    def _extract_from(zf: zipfile.ZipFile) -> int:
        n = 0
        for name in zf.namelist():
            base = name.replace("\\", "/").split("/")[-1]
            if _is_subject_pkl(base):
                (dest_dir / base).write_bytes(zf.read(name))
                n += 1
        return n

    with zipfile.ZipFile(outer_zip) as zf:
        extracted = _extract_from(zf)
        if extracted == 0:
            for name in zf.namelist():
                if name.lower().endswith(".zip"):
                    with zipfile.ZipFile(BytesIO(zf.read(name))) as inner:
                        extracted += _extract_from(inner)
    if extracted == 0:
        raise FileNotFoundError(
            f"no S<int>.pkl subject files found in {outer_zip} (unexpected archive layout)"
        )
    return extracted


def _ensure_extracted(
    data_dir: str | Path, archive_sha256: str | None = None
) -> Path:
    """Download + extract WESAD on first use; return the directory of SX.pkl files.

    The raw archive is large (~2.5 GB) and gitignored (data/raw/*). It is
    non-commercial + non-redistributable, so it must never be committed. Later
    loads reuse the extracted subject files."""
    root = Path(data_dir) / "WESAD"
    pkl_dir = root / "subjects"
    if sorted(pkl_dir.glob("S*.pkl")):
        return pkl_dir
    root.mkdir(parents=True, exist_ok=True)
    outer = root / "wesad.zip"
    if not (outer.exists() and outer.stat().st_size > 1_000_000):
        urllib.request.urlretrieve(DATA_URL, outer)
    if archive_sha256:
        actual = _sha256(outer)
        if actual != archive_sha256:
            raise ValueError(
                f"WESAD archive checksum mismatch for {outer}: "
                f"expected {archive_sha256}, got {actual}"
            )
    _extract_pkls(outer, pkl_dir)
    return pkl_dir


def load_wesad(
    data_dir: str | Path = "data/raw",
    metadata_dir: str | Path = "data/metadata",
    *,
    window: int = WINDOW,
    split_seed: int = SPLIT_SEED,
    n_test_subjects: int = N_TEST_SUBJECTS,
    normalize: str = NORMALIZE,
    downsample_hz: float = DOWNSAMPLE_HZ,
    overlap: float = OVERLAP,
    channels: list[str] | None = None,
    modality: str = MODALITY,
    exclude_subjects: list[int] | None = None,
    archive_sha256: str | None = None,
):
    """Return (pool, test) as SubjectSplits.

    pool = training subjects used to build subject-count curves.
    test = fixed held-out subjects (K = ``n_test_subjects``), subject-disjoint.

    The preprocessing knobs (``window``, ``split_seed``, ``n_test_subjects``,
    ``normalize``, ``downsample_hz``, ``overlap``, ``channels``,
    ``exclude_subjects``) are config-driven (dataset_params); the defaults
    reproduce the pre-registered setup (2100-sample windows @70 Hz, seed-0 K=5
    split, 5-ch chest physiology, per-window z-norm). ``modality`` is fixed to
    'chest' (only the chest physiology is implemented). ``archive_sha256``
    optionally pins the downloaded archive's hash."""
    from signal_aug.data.subject import SubjectSplits

    if modality != MODALITY:
        raise ValueError(
            f"WESAD loader only implements modality={MODALITY!r} "
            f"(chest physiology); got {modality!r}"
        )
    channels = channels or DEFAULT_CHANNELS
    pkl_dir = _ensure_extracted(data_dir, archive_sha256=archive_sha256)

    values, labels, subjects = build_arrays(
        pkl_dir, channels, set(exclude_subjects or [])
    )
    factor = downsample_factor(downsample_hz)
    X, y, subj = make_windows(
        values, labels, subjects,
        window=window, downsample_factor=factor, normalize=normalize, overlap=overlap,
    )

    pool_subjects, test_subjects = _split_subjects(
        subj, n_test_subjects=n_test_subjects, split_seed=split_seed
    )
    overlap_subjects = set(pool_subjects) & set(test_subjects)
    if overlap_subjects:
        raise ValueError(f"WESAD pool/test subject overlap: {overlap_subjects}")

    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)

    _record_metadata(
        Path(metadata_dir), X, y, subj, pool_subjects, test_subjects,
        window=window, split_seed=split_seed, n_test_subjects=n_test_subjects,
        normalize=normalize, downsample_hz=downsample_hz, overlap=overlap,
        channels=channels,
    )

    pool = SubjectSplits("WESAD_pool", X[pool_mask], y[pool_mask], subj[pool_mask])
    test = SubjectSplits("WESAD_test", X[test_mask], y[test_mask], subj[test_mask])
    return pool, test


# Config-relevant fields whose change means the recorded window set is stale.
_METADATA_DRIFT_FIELDS = (
    "window_length", "split_seed", "n_test_subjects", "normalize",
    "downsample_hz", "overlap", "channels",
    "n_pool_windows", "n_test_windows",
    "pool_windows_checksum", "test_windows_checksum",
)


def _record_metadata(
    metadata_dir: Path, X, y, subj, pool_subjects, test_subjects,
    *, window: int, split_seed: int, n_test_subjects: int, normalize: str,
    downsample_hz: float, overlap: float, channels: list[str],
) -> dict:
    """Write data/metadata/wesad.json on first build; afterwards verify.

    Records the license (non-commercial + attribution), DOI, config knobs, actual
    subject ids, per-subject class support, and a content hash of the pool/test
    window arrays. Only checksums + metadata are stored -- never the raw,
    non-redistributable signals (design 1.5). On a subsequent load the freshly
    built metadata is compared with the recorded file: identical -> left in place,
    drifted -> raise (spec section 7)."""
    from signal_aug.data.subject import windows_checksum

    metadata_dir.mkdir(parents=True, exist_ok=True)
    pool_mask = np.isin(subj, pool_subjects)
    test_mask = np.isin(subj, test_subjects)
    factor = downsample_factor(downsample_hz)
    meta = {
        "dataset": "WESAD",
        "license": LICENSE,
        "doi": DOI,
        "source": SOURCE,
        "url": DATA_URL,
        "redistribution": "not granted (raw data never committed; checksums + metadata only)",
        "modality": MODALITY,
        "channels": list(channels),
        "native_hz": NATIVE_HZ,
        "downsample_hz": int(downsample_hz),
        "downsample_factor": int(factor),
        "effective_hz": round(NATIVE_HZ / factor, 3),
        "anti_alias": "scipy.signal.decimate (zero-phase IIR Chebyshev-I) before decimation",
        "window_length": int(window),
        "overlap": float(overlap),
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "label_ids": LABEL_IDS,
        "preprocessing": (
            "chest physiology 5ch; labels {1,2,3} only (baseline/stress/amusement), "
            "{0,4,5,6,7} dropped; anti-aliased 700->"
            f"{round(NATIVE_HZ / factor, 1)} Hz decimation; "
            f"{window}-sample windows (overlap={overlap}) within (subject,label) "
            f"runs; NaN windows dropped; normalize={normalize}"
        ),
        "normalize": normalize,
        "split_seed": int(split_seed),
        "n_test_subjects": int(n_test_subjects),
        "pool_subjects": [int(s) for s in pool_subjects],
        "test_subjects": [int(s) for s in test_subjects],
        "subject_ids": [int(s) for s in sorted(np.unique(subj))],
        "n_pool_windows": int(pool_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "pool_windows_checksum": windows_checksum(X[pool_mask]),
        "test_windows_checksum": windows_checksum(X[test_mask]),
        "per_subject_class_support": per_subject_class_support(y, subj),
    }
    path = metadata_dir / "wesad.json"
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
                    f"WESAD dataset metadata drift vs {path}: {drift}. The recorded "
                    "window set no longer matches the freshly built one. If this "
                    "reconfiguration is intended, delete the stale metadata file to "
                    "regenerate it."
                )
            return meta  # matches the first-written record; leave it untouched
    path.write_text(json.dumps(meta, indent=2))
    return meta
