#!/usr/bin/env python
"""Generate before/after examples for the augmentation tab (issue #30).

Each method is applied to the SAME real training sample with the SAME
configured parameters as the experiments (config/augmentations.yaml), using the
production augmenters — so the figures show what the code actually does, not a
hand-drawn illustration. Writes report/assets/data/augmentation_examples.json,
which is committed so the report builds without the raw data present.

Policy: the source sample comes from a training split only (spec section 8).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from signal_aug.augmentations.methods import apply_augmentation

MAX_POINTS = 240
SEED = 0
OUT_PATH = Path("report/assets/data/augmentation_examples.json")
# GunPoint: short, single-channel, visually legible motion-capture traces, and
# one of the Phase 1 datasets, so readers see a curve the study actually used.
SOURCE_DATASET = "GunPoint"


def _decimate(series: np.ndarray, max_points: int = MAX_POINTS) -> list[float]:
    if len(series) <= max_points:
        thinned = series
    else:
        thinned = series[:: int(np.ceil(len(series) / max_points))]
    return [round(float(v), 4) for v in thinned]


def build_examples(dataset: str, data_dir: str) -> dict:
    from signal_aug.data.loader import load_dataset

    datasets_cfg = yaml.safe_load(Path("config/datasets.yaml").read_text(encoding="utf-8"))
    aug_cfg = yaml.safe_load(Path("config/augmentations.yaml").read_text(encoding="utf-8"))["augmentations"]
    data = load_dataset(dataset, datasets_cfg, data_dir=data_dir)

    X, y = data.X_train, data.y_train
    n_original = len(X)
    # a fixed, reproducible source sample: the medoid of class 0
    members = np.flatnonzero(y == 0)
    block = X[members, 0, :]
    source_idx = int(members[int(np.argmin(np.linalg.norm(block - block.mean(axis=0), axis=1)))])

    examples = {}
    for name, spec in aug_cfg.items():
        method, params = spec["method"], spec.get("params", {})
        X_aug, y_aug = apply_augmentation(method, X, y, seed=SEED, params=params)
        synthetic = X_aug[n_original:]
        entry = {
            "method": method,
            "params": params,
            "n_original": int(n_original),
            "n_after": int(len(X_aug)),
            "n_synthetic": int(len(synthetic)),
            "source": _decimate(X[source_idx, 0, :]),
            "synthetic": None,
            "label_changed": None,
        }
        if len(synthetic):
            # pick the synthetic sample closest to the source so the pair is
            # comparable; for methods that copy, this lands on the copy itself
            dists = np.linalg.norm(synthetic[:, 0, :] - X[source_idx, 0, :], axis=1)
            pick = int(np.argmin(dists))
            entry["synthetic"] = _decimate(synthetic[pick, 0, :])
            entry["synthetic_label"] = int(y_aug[n_original + pick])
            entry["source_label"] = int(y[source_idx])
            # label_shuffle is the only method that may relabel: report the rate
            src_labels = y_aug[n_original:]
            entry["label_changed"] = bool(entry["synthetic_label"] != entry["source_label"])
            entry["n_classes_in_synthetic"] = int(len(np.unique(src_labels)))
        examples[name] = entry
        print(f"[aug] {name}: {n_original} -> {len(X_aug)} samples")

    return {
        "source_dataset": dataset,
        "source_index": source_idx,
        "source_class": int(y[source_idx]),
        "seed": SEED,
        "max_points": MAX_POINTS,
        "note": "学習split の1サンプルに、実験と同じ config のパラメータで各拡張を適用した実出力",
        "methods": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=SOURCE_DATASET)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    payload = build_examples(args.dataset, args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(payload['methods'])} methods -> {out}")


if __name__ == "__main__":
    main()
