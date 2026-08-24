#!/usr/bin/env python
"""Per-dataset summary of how each augmentation changes the training
distribution (issue: 0826 discussion prep for the 学習と評価 tab).

For every dataset whose training-side signal values are already committed to
report/assets/data/dataset_samples.json (the same datasets described in the
dataset tab), this runs the SAME production augmenters with the SAME
parameters as config/augmentations.yaml against that dataset's real training
split, and records two compact, honest measures of what changed:

  * class balance shift: total-variation distance between the class
    proportions before and after augmentation. 0 = unchanged, up to 1 =
    completely different mix. Most methods should be ~0 (they draw synthetic
    samples in proportion to the existing classes); label_shuffle is the
    exception, since its synthetic portion gets uniformly random labels.
  * value spread shift: ratio of the augmented pool's standard deviation to
    the original's (channel 0). >1 means the augmented pool is more spread
    out (jitter/scaling are expected to do this); ~1 means little change.

Writes report/assets/data/distribution_shift.json. PAMAP2 and WESAD are
skipped for the same reason their waveforms are not committed (cost/licence,
see build_dataset_samples.py) -- there is no local raw data to augment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from signal_aug.augmentations.methods import apply_augmentation

OUT_PATH = Path("report/assets/data/distribution_shift.json")
SEED = 0


def _class_proportions(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return counts / max(counts.sum(), 1)


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _dataset_shift(X: np.ndarray, y: np.ndarray, aug_cfg: dict) -> dict:
    n_classes = int(y.max()) + 1
    before_props = _class_proportions(y, n_classes)
    before_std = float(X[:, 0, :].std())
    before_mean = float(X[:, 0, :].mean())
    n_before = len(y)

    methods = {}
    for name, spec in aug_cfg.items():
        if name == "none":
            continue
        method, params = spec["method"], spec.get("params", {})
        X_aug, y_aug = apply_augmentation(method, X, y, seed=SEED, params=params)
        after_props = _class_proportions(y_aug, n_classes)
        after_std = float(X_aug[:, 0, :].std())
        after_mean = float(X_aug[:, 0, :].mean())
        methods[name] = {
            "n_synthetic": int(len(y_aug) - n_before),
            "class_balance_shift": round(_tv_distance(before_props, after_props), 4),
            "std_before": round(before_std, 4),
            "std_after": round(after_std, 4),
            "std_ratio": round(after_std / before_std, 4) if before_std > 1e-9 else None,
            "mean_shift": round(after_mean - before_mean, 4),
        }
    return {
        "n_before": n_before,
        "n_classes": n_classes,
        "class_proportions_before": [round(float(p), 4) for p in before_props],
        "std_before": round(before_std, 4),
        "methods": methods,
    }


def build_ucr_shifts(datasets_cfg: dict, data_dir: str, aug_cfg: dict, only: set | None) -> dict:
    from signal_aug.data.loader import load_dataset

    out = {}
    names = [n for n, s in datasets_cfg["datasets"].items() if s.get("source") == "ucr"]
    for name in sorted(names):
        if only is not None and name not in only:
            continue
        data = load_dataset(name, datasets_cfg, data_dir=data_dir)
        out[name] = _dataset_shift(data.X_train, data.y_train, aug_cfg)
        print(f"[ucr] {name}: done ({len(data.y_train)} train samples)")
    return out


def build_subject_shifts(aug_cfg: dict, only: set | None) -> dict:
    from signal_aug.data.subject_datasets import load_subject_dataset

    out = {}
    for name in ("UCI_HAR", "WISDM"):
        if only is not None and name not in only:
            continue
        pool, _test = load_subject_dataset(name)
        out[name] = _dataset_shift(pool.X, pool.y, aug_cfg)
        print(f"[subject] {name}: done ({len(pool.y)} pool samples)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--only", nargs="*", default=None,
                        help="limit to these dataset keys (default: all with local samples)")
    args = parser.parse_args()

    datasets_cfg = yaml.safe_load(Path("config/datasets.yaml").read_text(encoding="utf-8"))
    aug_cfg = yaml.safe_load(Path("config/augmentations.yaml").read_text(encoding="utf-8"))["augmentations"]
    only = set(args.only) if args.only else None

    shifts = {}
    shifts.update(build_ucr_shifts(datasets_cfg, args.data_dir, aug_cfg, only))
    shifts.update(build_subject_shifts(aug_cfg, only))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shifts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(shifts)} datasets -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
