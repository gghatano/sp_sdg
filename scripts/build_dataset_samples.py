#!/usr/bin/env python
"""Extract per-dataset sample waveforms and raw-value excerpts for the report's
dataset tab (issue #29).

Writes report/assets/data/dataset_samples.json, which is committed so that the
report builds without the raw data present (raw data is gitignored; PAMAP2 and
WESAD are additionally excluded here, see below).

Policy:
  * samples are drawn from the TRAINING/pool split only, never from test
    (spec section 8: test data is not used for anything, display included).
  * PAMAP2 is skipped (1.6 GB download for a figure) and WESAD is skipped
    (license grants no redistribution, so no excerpt of its signal is
    committed). Both are still described in the report from
    data/metadata/*.json, which carries no signal values.
  * waveforms are decimated to at most MAX_POINTS points so the JSON stays
    small; the decimation is stride-based (no smoothing) to keep the shape
    honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

MAX_POINTS = 240        # per rendered waveform trace
N_EXCERPT_ROWS = 3      # rows in the "sample values" table
N_EXCERPT_COLS = 8      # timepoints shown per row
OUT_PATH = Path("report/assets/data/dataset_samples.json")

# Subject datasets whose signal values may be committed. PAMAP2/WESAD are
# excluded on cost/licence grounds (see module docstring).
SUBJECT_SAMPLE_DATASETS = ["UCI_HAR", "WISDM"]


def _decimate(series: np.ndarray, max_points: int = MAX_POINTS) -> list[float]:
    """Stride-decimate a 1-D trace to <= max_points, keeping the first sample."""
    if len(series) <= max_points:
        thinned = series
    else:
        stride = int(np.ceil(len(series) / max_points))
        thinned = series[::stride]
    return [round(float(v), 4) for v in thinned]


def _class_traces(X: np.ndarray, y: np.ndarray, class_names: list[str], channel: int = 0) -> list[dict]:
    """One representative trace per class: the medoid of the class (the sample
    closest to the class mean), which is more typical than an arbitrary first
    sample."""
    traces = []
    for idx, name in enumerate(class_names):
        members = np.flatnonzero(y == idx)
        if len(members) == 0:
            continue
        block = X[members, channel, :]
        centre = block.mean(axis=0)
        medoid = members[int(np.argmin(np.linalg.norm(block - centre, axis=1)))]
        traces.append({
            "class_index": idx,
            "class_name": name,
            "n_samples": int(len(members)),
            "sample_index": int(medoid),
            "values": _decimate(X[medoid, channel, :]),
        })
    return traces


def _channel_traces(X: np.ndarray, y: np.ndarray, class_names: list[str],
                    channels: list[str], class_index: int = 0) -> list[dict]:
    """All channels of one representative sample, so multi-channel datasets show
    what a single window actually looks like across sensors."""
    members = np.flatnonzero(y == class_index)
    if len(members) == 0:
        return []
    block = X[members, 0, :]
    medoid = members[int(np.argmin(np.linalg.norm(block - block.mean(axis=0), axis=1)))]
    return [
        {"channel": ch, "values": _decimate(X[medoid, c, :])}
        for c, ch in enumerate(channels[: X.shape[1]])
    ]


def _excerpt(X: np.ndarray, channel: int = 0) -> list[list[float]]:
    """First few timepoints of the first few training samples: the 'what does a
    row actually contain' table."""
    return [
        [round(float(v), 3) for v in X[i, channel, :N_EXCERPT_COLS]]
        for i in range(min(N_EXCERPT_ROWS, len(X)))
    ]


def _class_distribution(y: np.ndarray, class_names: list[str]) -> list[dict]:
    return [
        {"class_name": name, "count": int(np.sum(y == idx))}
        for idx, name in enumerate(class_names)
    ]


def build_ucr_profiles(datasets_cfg: dict, data_dir: str, skip_missing: bool = False) -> dict:
    from signal_aug.data.loader import load_dataset

    profiles = {}
    names = [n for n, s in datasets_cfg["datasets"].items() if s.get("source") == "ucr"]
    for name in sorted(names):
        try:
            data = load_dataset(name, datasets_cfg, data_dir=data_dir)
        except Exception as exc:  # noqa: BLE001 - the archive host is flaky
            if not skip_missing:
                raise
            print(f"[skip] {name}: {type(exc).__name__} ({exc})")
            continue
        profiles[name] = {
            "kind": "ucr",
            "n_train": int(len(data.y_train)),
            "n_test": int(len(data.y_test)),
            "n_channels": int(data.X_train.shape[1]),
            "length": int(data.X_train.shape[2]),
            "n_classes": len(data.class_names),
            "class_names": list(data.class_names),
            "class_distribution": _class_distribution(data.y_train, data.class_names),
            "traces": _class_traces(data.X_train, data.y_train, data.class_names),
            "excerpt": _excerpt(data.X_train),
            "excerpt_note": "train split, channel 0, 先頭3サンプル×先頭8時点(z正規化後)",
        }
        print(f"[ucr] {name}: {len(data.y_train)} train, {data.X_train.shape[2]} points, "
              f"{len(data.class_names)} classes")
    return profiles


def _subject_class_names(name: str, meta: dict, n_classes: int, data_dir: str) -> list[str]:
    """SubjectSplits carries no class names, so recover them from the dataset's
    own metadata (WISDM records its activity list) or, for UCI HAR, from the
    activity_labels.txt shipped inside the raw archive. Falls back to indices."""
    if meta.get("activities"):
        return list(meta["activities"])
    if meta.get("class_names"):
        return list(meta["class_names"])
    labels_file = next(Path(data_dir).glob(f"{name}/**/activity_labels.txt"), None)
    if labels_file:
        names = [line.split(maxsplit=1)[1] for line in
                 labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) == n_classes:
            return names
    return [str(i) for i in range(n_classes)]


def build_subject_profiles(data_dir: str = "data/raw") -> dict:
    from signal_aug.data.subject_datasets import load_subject_dataset

    profiles = {}
    for name in SUBJECT_SAMPLE_DATASETS:
        pool, _test = load_subject_dataset(name)
        meta = json.loads(Path(f"data/metadata/{name.lower()}.json").read_text(encoding="utf-8"))
        channels = meta.get("channels", [])
        n_classes = int(pool.y.max()) + 1
        class_names = _subject_class_names(name, meta, n_classes, data_dir)
        profiles[name] = {
            "kind": "subject",
            "n_subjects": int(pool.n_subjects()),
            "n_pool_windows": int(len(pool.y)),
            "n_channels": int(pool.X.shape[1]),
            "length": int(pool.X.shape[2]),
            "n_classes": len(class_names),
            "class_names": class_names,
            "class_distribution": _class_distribution(pool.y, class_names),
            "traces": _class_traces(pool.X, pool.y, class_names),
            "channel_traces": _channel_traces(pool.X, pool.y, class_names, channels),
            "channel_trace_class": class_names[0] if class_names else None,
            "excerpt": _excerpt(pool.X),
            "excerpt_note": "pool(学習側)split、チャネル0、先頭3窓×先頭8時点(窓ごとz正規化後)",
        }
        print(f"[subject] {name}: {len(pool.y)} pool windows, {pool.X.shape[1]}ch x "
              f"{pool.X.shape[2]}, {len(class_names)} classes")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--skip-subject", action="store_true",
                        help="only rebuild the UCR entries (subject loaders download data)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="skip datasets that fail to load instead of aborting "
                             "(the UCR host is intermittently unavailable)")
    parser.add_argument("--merge", action="store_true",
                        help="merge into the existing output instead of replacing it, so a "
                             "partial rebuild does not drop previously generated datasets")
    args = parser.parse_args()

    datasets_cfg = yaml.safe_load(Path("config/datasets.yaml").read_text(encoding="utf-8"))
    out = Path(args.out)
    samples = {"max_points": MAX_POINTS, "datasets": {}}
    if args.merge and out.exists():
        samples = json.loads(out.read_text(encoding="utf-8"))
        samples["max_points"] = MAX_POINTS
    samples["datasets"].update(build_ucr_profiles(datasets_cfg, args.data_dir, args.skip_missing))
    if not args.skip_subject:
        samples["datasets"].update(build_subject_profiles(args.data_dir))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(samples['datasets'])} datasets -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
