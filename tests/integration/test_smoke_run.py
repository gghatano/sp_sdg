"""Integration: full runner path on synthetic data - data load, augmentation,
training, prediction, metrics, manifest, resume (spec 3.6: integration scope).
Network-free and fast (tiny models).
"""

import json
from pathlib import Path

import pytest
import yaml

from signal_aug.experiments.manifest import validate_manifest
from signal_aug.experiments.runner import run_experiment


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    """Run the smoke experiment once into an isolated runs/ directory."""
    runs_dir = tmp_path_factory.mktemp("runs")
    results = run_experiment(
        "config/experiments/smoke.yaml",
        config_dir="config",
        runs_dir=runs_dir,
        repo_root=".",
    )
    return runs_dir, results


def test_all_runs_complete(smoke_result):
    _, results = smoke_result
    assert len(results) == 4  # 1 dataset x 2 augs x 2 models x 1 seed
    assert all(r["status"] == "completed" for r in results)


def test_manifests_valid_and_artifacts_exist(smoke_result):
    runs_dir, results = smoke_result
    for manifest in results:
        assert validate_manifest(manifest) == []
        metrics = json.loads(Path(manifest["metrics_path"]).read_text())
        assert 0.0 <= metrics["accuracy"] <= 1.0
        pred_lines = Path(manifest["predictions_path"]).read_text().strip().splitlines()
        assert len(pred_lines) - 1 == metrics["n_test"]
        assert Path(manifest["log_path"]).exists()


def test_models_learn_signal(smoke_result):
    """Synthetic sine-vs-square is easy; every model should beat chance clearly."""
    _, results = smoke_result
    for manifest in results:
        metrics = json.loads(Path(manifest["metrics_path"]).read_text())
        assert metrics["accuracy"] > 0.6, manifest["run_id"]


def test_resume_skips_completed(smoke_result, capsys):
    runs_dir, _ = smoke_result
    results = run_experiment(
        "config/experiments/smoke.yaml",
        config_dir="config",
        runs_dir=runs_dir,
        repo_root=".",
    )
    captured = capsys.readouterr()
    assert captured.out.count("[skip]") == 4
    assert all(r["status"] == "completed" for r in results)


def test_draft_config_rejected(tmp_path):
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump({"status": "draft", "phase": 2, "name": "x",
                                     "datasets": [], "augmentations": [], "models": [], "seeds": []}))
    with pytest.raises(ValueError, match="draft"):
        run_experiment(draft, config_dir="config", runs_dir=tmp_path / "runs")
