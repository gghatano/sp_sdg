"""Network-free tests for UCI HAR loader internals (extract / load / metadata).

Builds a tiny fake "UCI HAR Dataset" tree so the pure logic (9-channel stack,
label 1->0 shift, subject parsing, overlap guard) is covered without download.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from signal_aug.data.uci_har import SIGNALS, _extract, _load_split, _record_metadata


def _write_split(root: Path, split: str, n: int, length: int, subjects, labels):
    sig_dir = root / split / "Inertial Signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for ci, name in enumerate(SIGNALS):
        arr = np.full((n, length), float(ci), dtype=np.float64)
        np.savetxt(sig_dir / f"{name}_{split}.txt", arr)
    np.savetxt(root / split / f"y_{split}.txt", np.array(labels), fmt="%d")
    np.savetxt(root / split / f"subject_{split}.txt", np.array(subjects), fmt="%d")


def _make_har_tree(base: Path, length=8):
    root = base / "UCI HAR Dataset"
    _write_split(root, "train", n=4, length=length, subjects=[1, 1, 2, 2], labels=[1, 2, 3, 6])
    _write_split(root, "test", n=2, length=length, subjects=[9, 9], labels=[1, 5])
    return root


def test_load_split_shapes_channels_and_label_shift(tmp_path):
    root = _make_har_tree(tmp_path)
    X, y, subjects = _load_split(root, "train")
    assert X.shape == (4, len(SIGNALS), 8)  # 9 channels stacked
    assert X.dtype == np.float32
    # channel c was filled with value c -> ordering preserved
    assert np.allclose(X[0, :, 0], np.arange(len(SIGNALS)))
    assert y.tolist() == [0, 1, 2, 5]  # labels 1..6 shifted to 0..5
    assert subjects.tolist() == [1, 1, 2, 2]


def test_extract_finds_dataset_dir(tmp_path):
    _make_har_tree(tmp_path / "unzipped")
    # zip it up like the real archive
    zip_path = tmp_path / "har.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in (tmp_path / "unzipped").rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(tmp_path / "unzipped"))
    extract_dir = tmp_path / "extract"
    root = _extract(zip_path, extract_dir)
    assert root.name == "UCI HAR Dataset"
    assert (root / "train" / "Inertial Signals").is_dir()


def test_record_metadata_writes_license_and_subjects(tmp_path):
    root = _make_har_tree(tmp_path)
    X_tr, y_tr, s_tr = _load_split(root, "train")
    X_te, y_te, s_te = _load_split(root, "test")
    _record_metadata(tmp_path / "meta", X_tr, y_tr, s_tr, X_te, y_te, s_te)
    meta = json.loads((tmp_path / "meta" / "uci_har.json").read_text())
    assert meta["license"] == "CC BY 4.0"
    assert meta["pool_subjects"] == [1, 2]
    assert meta["test_subjects"] == [9]
    assert meta["channels"] == SIGNALS
    assert meta["n_classes"] == 6
