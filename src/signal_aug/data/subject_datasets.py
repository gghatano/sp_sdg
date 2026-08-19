"""Dispatch registry for subject-labelled datasets used by the subject-count
reduction study (Phase 4-5). Experiment configs reference a dataset by key; the
runner resolves it here instead of importing a single loader (issue #21 DS-1).

Each loader returns (pool, test) as SubjectSplits, with pool/test being
subject-disjoint (spec section 8). New subject datasets register one entry.
"""

from __future__ import annotations

from typing import Callable

from signal_aug.data.pamap2 import load_pamap2
from signal_aug.data.uci_har import load_uci_har
from signal_aug.data.wesad import load_wesad
from signal_aug.data.wisdm import load_wisdm

SUBJECT_LOADERS: dict[str, Callable] = {
    "UCI_HAR": load_uci_har,
    "WISDM": load_wisdm,
    "PAMAP2": load_pamap2,
    "WESAD": load_wesad,
}


def load_subject_dataset(dataset: str, **kwargs):
    """Resolve a subject dataset key to its (pool, test) SubjectSplits."""
    try:
        loader = SUBJECT_LOADERS[dataset]
    except KeyError:
        known = ", ".join(sorted(SUBJECT_LOADERS))
        raise ValueError(f"unknown subject dataset '{dataset}'; known: {known}") from None
    return loader(**kwargs)
