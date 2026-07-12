"""Subject-count grid expansion and pre-registration guard (Phase 4)."""

import pytest

from signal_aug.experiments.subject_runner import expand_subject_grid, run_subject_experiment

EXP = {
    "phase": 4,
    "name": "subject_count",
    "dataset": "UCI_HAR",
    "model": "cnn1d",
    "subject_counts": [3, 6],
    "repeats": 2,
    "augmentations": ["none", "mixup"],
    "pre_registered": True,
}
AUG = {"augmentations": {"none": {"method": "none"}, "mixup": {"method": "mixup", "params": {"ratio": 1.0}}}}
MODEL = {"models": {"cnn1d": {"type": "cnn1d", "params": {"epochs": 5}}}}


def test_subject_grid_size_and_ids():
    runs = expand_subject_grid(EXP, AUG, MODEL)
    assert len(runs) == 2 * 2 * 2  # counts x repeats x augs
    ids = {r.run_id for r in runs}
    assert "subject_count_UCI_HAR_n3_none_cnn1d_r0" in ids
    assert "subject_count_UCI_HAR_n6_mixup_cnn1d_r1" in ids
    assert all(r.subject_count in (3, 6) for r in runs)


def test_subject_grid_carries_params():
    runs = expand_subject_grid(EXP, AUG, MODEL)
    mixup = next(r for r in runs if r.augmentation == "mixup")
    assert mixup.augmentation_params == {"ratio": 1.0}
    assert mixup.model_params == {"epochs": 5}


def test_run_rejects_unregistered_config(tmp_path):
    import yaml
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(dict(EXP, pre_registered=False)))
    with pytest.raises(ValueError, match="pre-registered"):
        run_subject_experiment(cfg, runs_dir=tmp_path / "runs")
