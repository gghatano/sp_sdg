"""Subject-aware data handling for the subject-count reduction study
(Phase 4-5). The central leakage rule (spec section 8): a subject's data must
never straddle the train and test splits, and subject-count learning curves
must add whole subjects at a time — never individual samples.

This module is dataset-agnostic: it operates on arrays plus a per-sample
subject-id vector, so UCI HAR / WISDM / any subject-tagged dataset reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SubjectSplits:
    """Signals plus a per-sample subject id. Shapes mirror DatasetSplits:
    X is (n, channels, length) float32, y is (n,) int64, subjects is (n,)."""

    name: str
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray

    def unique_subjects(self) -> np.ndarray:
        return np.unique(self.subjects)

    def n_subjects(self) -> int:
        return len(self.unique_subjects())


def subject_holdout(
    data: SubjectSplits, test_subjects: list, val_subjects: list | None = None
) -> dict[str, np.ndarray]:
    """Partition by subject id. Any subject in test/val is fully excluded from
    train. Raises if a subject is assigned to more than one split (a leak)."""
    test_set = set(np.asarray(test_subjects).tolist())
    val_set = set(np.asarray(val_subjects).tolist()) if val_subjects else set()
    if test_set & val_set:
        raise ValueError(f"subjects in both test and val: {test_set & val_set}")

    masks = {
        "test": np.isin(data.subjects, list(test_set)),
        "val": np.isin(data.subjects, list(val_set)) if val_set else np.zeros(len(data.y), bool),
    }
    masks["train"] = ~(masks["test"] | masks["val"])

    out = {}
    for split, mask in masks.items():
        out[f"X_{split}"] = data.X[mask]
        out[f"y_{split}"] = data.y[mask]
        out[f"subjects_{split}"] = data.subjects[mask]

    # defensive: assert disjoint subject sets (the leakage guard, spec section 8)
    train_subj = set(out["subjects_train"].tolist())
    for other in ("test", "val"):
        overlap = train_subj & set(out[f"subjects_{other}"].tolist())
        if overlap:
            raise ValueError(f"subject leak between train and {other}: {overlap}")
    return out


def select_train_subjects(
    train_subjects: np.ndarray, n_subjects: int, seed: int
) -> list:
    """Deterministically pick n whole subjects from the training pool for one
    point on the subject-count learning curve. Adds subjects, never samples."""
    unique = np.unique(train_subjects)
    if n_subjects > len(unique):
        raise ValueError(f"requested {n_subjects} subjects but only {len(unique)} available")
    rng = np.random.default_rng([seed, n_subjects])
    chosen = rng.choice(unique, size=n_subjects, replace=False)
    return sorted(chosen.tolist())


def subset_by_subjects(data_dict: dict[str, np.ndarray], keep_subjects: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the training samples belonging to keep_subjects."""
    mask = np.isin(data_dict["subjects_train"], list(keep_subjects))
    return data_dict["X_train"][mask], data_dict["y_train"][mask]


def make_synthetic_subjects(
    n_subjects: int = 12, samples_per_subject: int = 20, length: int = 48, n_classes: int = 2, seed: int = 0
) -> SubjectSplits:
    """Synthetic subject-tagged dataset for tests. Each subject has an
    individual offset/scale so that subject leakage would measurably inflate
    accuracy — making leak-prevention testable."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length, dtype=np.float32)
    X, y, subjects = [], [], []
    for s in range(n_subjects):
        subj_offset = rng.uniform(-1, 1)
        subj_scale = rng.uniform(0.8, 1.2)
        for _ in range(samples_per_subject):
            cls = int(rng.integers(0, n_classes))
            phase = rng.uniform(0, 2 * np.pi)
            base = np.sin(t + phase) if cls == 0 else np.sign(np.sin(t + phase))
            sig = subj_scale * base + subj_offset + rng.normal(0, 0.2, size=length)
            X.append(sig.astype(np.float32)[None, :])
            y.append(cls)
            subjects.append(s)
    return SubjectSplits(
        "synthetic_subjects",
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(subjects),
    )
