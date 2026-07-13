"""End-to-end manifests -> results.json -> audit (spec 3.6 integration scope).

Runs the network-free smoke grid, then exercises the aggregation and audit
layers that test_smoke_run.py does not reach.
"""

import json
from pathlib import Path

import pytest

from signal_aug.evaluation.audit import audit_all, audit_run
from signal_aug.experiments.runner import run_experiment
from signal_aug.reporting.aggregate import build_results_json


@pytest.fixture(scope="module")
def smoke_runs_dir(tmp_path_factory):
    runs_dir = tmp_path_factory.mktemp("runs")
    run_experiment("config/experiments/smoke.yaml", config_dir="config", runs_dir=runs_dir, repo_root=".")
    return runs_dir


def test_audit_all_passes_for_clean_runs(smoke_runs_dir):
    report = audit_all(smoke_runs_dir / "manifests")
    assert report["n_runs"] == 4
    assert report["n_passed"] == 4
    assert report["n_problem_runs"] == 0


def test_build_results_json_summarizes(smoke_runs_dir, tmp_path):
    out = tmp_path / "results.json"
    data = build_results_json(
        manifests_dir=smoke_runs_dir / "manifests",
        audit_path=tmp_path / "missing_audit.json",
        output_path=out,
        config_dir="config",
    )
    assert out.exists()
    assert len(data["runs"]) == 4
    assert all(r["status"] == "completed" for r in data["runs"])
    # summary groups by (dataset, fraction, aug, model); smoke has 2 augs x 2 models
    assert len(data["summary"]) == 4
    for s in data["summary"]:
        assert 0.0 <= s["accuracy_mean"] <= 1.0
        assert s["n_seeds"] == 1


def test_audit_catches_prediction_row_count_mismatch(smoke_runs_dir):
    """Corrupt a predictions file and confirm the audit flags it."""
    manifests = sorted((smoke_runs_dir / "manifests").glob("*.json"))
    manifest = json.loads(manifests[0].read_text())
    pred = Path(manifest["predictions_path"])
    original = pred.read_text()
    try:
        # drop the last prediction row -> count no longer matches n_test
        lines = original.strip().splitlines()
        pred.write_text("\n".join(lines[:-1]) + "\n")
        result = audit_run(manifest)
        assert result["passed"] is False
        assert any("prediction rows" in p for p in result["problems"])
    finally:
        pred.write_text(original)


def test_audit_catches_out_of_range_metric(smoke_runs_dir):
    manifests = sorted((smoke_runs_dir / "manifests").glob("*.json"))
    manifest = json.loads(manifests[0].read_text())
    metrics_path = Path(manifest["metrics_path"])
    original = metrics_path.read_text()
    try:
        bad = json.loads(original)
        bad["accuracy"] = 1.5  # out of [0, 1]
        metrics_path.write_text(json.dumps(bad))
        result = audit_run(manifest)
        assert result["passed"] is False
        assert any("out of range" in p for p in result["problems"])
    finally:
        metrics_path.write_text(original)
