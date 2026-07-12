#!/usr/bin/env python
"""Download (and cache) the UCR datasets referenced in config/datasets.yaml.

Records checksums into data/metadata/checksums.json.
"""

import argparse
import json
from pathlib import Path

import yaml

from signal_aug.data.loader import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="subset of dataset names (default: all ucr)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("config/datasets.yaml").read_text())
    names = args.datasets or [n for n, s in cfg["datasets"].items() if s["source"] == "ucr"]

    metadata_path = Path("data/metadata/checksums.json")
    checksums = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    for name in names:
        print(f"[download] {name}")
        data = load_dataset(name, cfg)
        checksums[name] = {
            "dataset_checksum": data.dataset_checksum,
            "split_checksum": data.split_checksum,
            "n_train": len(data.y_train),
            "n_test": len(data.y_test),
            "n_channels": int(data.X_train.shape[1]),
            "length": int(data.X_train.shape[2]),
            "n_classes": len(data.class_names),
        }
        print(f"  train={len(data.y_train)} test={len(data.y_test)} length={data.X_train.shape[2]}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(checksums, indent=2))
    print(f"[done] checksums written to {metadata_path}")


if __name__ == "__main__":
    main()
