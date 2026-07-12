"""UCI HAR (Human Activity Recognition Using Smartphones) raw-signal loader
for the subject-count reduction study (Phase 4-5).

Loads the 9-channel inertial signals (body_acc / body_gyro / total_acc, x/y/z,
128 samples per 2.56 s window) with the per-window subject id. The official
train/test split is subject-disjoint (21 vs 9 subjects); we keep the 9 test
subjects as a fixed held-out set and build learning curves from the 21 train
subjects (spec section 8: subjects never straddle train and test).

License: CC BY 4.0. Source: Anguita et al. (2013), ESANN. UCI ML Repository
dataset 240 (DOI:10.24432/C54S4K). Recorded to data/metadata/ on download.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

LICENSE = "CC BY 4.0"
SOURCE = "Anguita et al. (2013), ESANN; UCI ML Repository dataset 240, DOI:10.24432/C54S4K"


def _download(dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists() and dest_zip.stat().st_size > 1_000_000:
        return
    urllib.request.urlretrieve(DATA_URL, dest_zip)


def _extract(zip_path: Path, extract_dir: Path) -> Path:
    # The archive nests a second zip in some mirrors; handle both layouts.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    inner = list(extract_dir.glob("**/UCI HAR Dataset"))
    if inner:
        return inner[0]
    # nested zip case
    nested = list(extract_dir.glob("*.zip"))
    for nz in nested:
        with zipfile.ZipFile(nz) as zf:
            zf.extractall(extract_dir)
    inner = list(extract_dir.glob("**/UCI HAR Dataset"))
    if not inner:
        raise FileNotFoundError("UCI HAR Dataset directory not found after extraction")
    return inner[0]


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig_dir = root / split / "Inertial Signals"
    channels = [np.loadtxt(sig_dir / f"{name}_{split}.txt", dtype=np.float32) for name in SIGNALS]
    X = np.stack(channels, axis=1)  # (n, 9, 128)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1  # 1..6 -> 0..5
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return X.astype(np.float32), y, subjects


def load_uci_har(data_dir: str | Path = "data/raw", metadata_dir: str | Path = "data/metadata"):
    """Return (pool, test) as SubjectSplits.

    pool = official train subjects (21) used to build subject-count curves.
    test = official test subjects (9), fixed held-out, subject-disjoint from pool.
    """
    from signal_aug.data.subject import SubjectSplits

    data_dir = Path(data_dir)
    har_dir = data_dir / "UCI_HAR"
    zip_path = har_dir / "uci_har.zip"
    _download(zip_path)
    root = _extract(zip_path, har_dir)

    X_tr, y_tr, s_tr = _load_split(root, "train")
    X_te, y_te, s_te = _load_split(root, "test")

    # sanity: subject sets must be disjoint (they are, by construction)
    overlap = set(s_tr.tolist()) & set(s_te.tolist())
    if overlap:
        raise ValueError(f"UCI HAR train/test subject overlap: {overlap}")

    _record_metadata(Path(metadata_dir), X_tr, y_tr, s_tr, X_te, y_te, s_te)

    pool = SubjectSplits("UCI_HAR_pool", X_tr, y_tr, s_tr)
    test = SubjectSplits("UCI_HAR_test", X_te, y_te, s_te)
    return pool, test


def _record_metadata(metadata_dir: Path, X_tr, y_tr, s_tr, X_te, y_te, s_te) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": "UCI_HAR",
        "license": LICENSE,
        "source": SOURCE,
        "url": DATA_URL,
        "channels": SIGNALS,
        "window_length": int(X_tr.shape[2]),
        "n_classes": int(max(y_tr.max(), y_te.max()) + 1),
        "pool_subjects": sorted(set(int(s) for s in s_tr)),
        "test_subjects": sorted(set(int(s) for s in s_te)),
        "n_pool_windows": int(len(y_tr)),
        "n_test_windows": int(len(y_te)),
    }
    (metadata_dir / "uci_har.json").write_text(json.dumps(meta, indent=2))
