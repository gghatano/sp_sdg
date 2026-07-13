"""Shared fixtures for the test suite."""

import numpy as np
import pytest

from signal_aug.data.loader import load_synthetic
from signal_aug.data.subject import make_synthetic_subjects

SYNTH_SPEC = {"n_train": 40, "n_test": 20, "length": 64, "n_channels": 1, "n_classes": 2}


@pytest.fixture
def synth_spec():
    return dict(SYNTH_SPEC)


@pytest.fixture
def synthetic_dataset():
    """Small deterministic two-class synthetic DatasetSplits."""
    return load_synthetic(SYNTH_SPEC)


@pytest.fixture
def subject_data():
    """Subject-tagged synthetic data (12 subjects) with per-subject offset/scale."""
    return make_synthetic_subjects(n_subjects=12, samples_per_subject=15, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
