"""H2 generalization-gap analysis (issue #12)."""

from signal_aug.evaluation.gap_analysis import analyze_gap, generalization_gap, sdg_effect


def _none_rows():
    # cnn1d overfits (large gap), minirocket does not (small gap)
    rows = []
    for seed in range(3):
        rows.append({"dataset": "D1", "model": "cnn1d", "augmentation": "none",
                     "status": "completed", "train_accuracy": 0.99, "accuracy": 0.80})
        rows.append({"dataset": "D2", "model": "cnn1d", "augmentation": "none",
                     "status": "completed", "train_accuracy": 0.95, "accuracy": 0.85})
        rows.append({"dataset": "D1", "model": "minirocket", "augmentation": "none",
                     "status": "completed", "train_accuracy": 0.92, "accuracy": 0.90})
        rows.append({"dataset": "D2", "model": "minirocket", "augmentation": "none",
                     "status": "completed", "train_accuracy": 0.91, "accuracy": 0.905})
    return rows


def _summary():
    # cnn1d benefits from aug; minirocket does not
    return [
        {"dataset": "D1", "model": "cnn1d", "augmentation": "none", "accuracy_mean": 0.80, "train_fraction": 1.0},
        {"dataset": "D1", "model": "cnn1d", "augmentation": "jitter", "accuracy_mean": 0.86, "train_fraction": 1.0},
        {"dataset": "D2", "model": "cnn1d", "augmentation": "none", "accuracy_mean": 0.85, "train_fraction": 1.0},
        {"dataset": "D2", "model": "cnn1d", "augmentation": "jitter", "accuracy_mean": 0.88, "train_fraction": 1.0},
        {"dataset": "D1", "model": "minirocket", "augmentation": "none", "accuracy_mean": 0.90, "train_fraction": 1.0},
        {"dataset": "D1", "model": "minirocket", "augmentation": "jitter", "accuracy_mean": 0.90, "train_fraction": 1.0},
        {"dataset": "D2", "model": "minirocket", "augmentation": "none", "accuracy_mean": 0.905, "train_fraction": 1.0},
        {"dataset": "D2", "model": "minirocket", "augmentation": "jitter", "accuracy_mean": 0.90, "train_fraction": 1.0},
    ]


def test_generalization_gap_averages_over_seeds():
    gaps = generalization_gap(_none_rows())
    assert abs(gaps[("D1", "cnn1d")] - 0.19) < 1e-9
    assert abs(gaps[("D1", "minirocket")] - 0.02) < 1e-9


def test_sdg_effect_is_mean_delta_vs_none():
    eff = sdg_effect(_summary(), train_fraction=1.0)
    assert abs(eff[("D1", "cnn1d")] - 0.06) < 1e-9      # 0.86 - 0.80
    assert abs(eff[("D2", "minirocket")] - (-0.005)) < 1e-9


def test_analyze_gap_contrasts_models():
    a = analyze_gap(_none_rows(), _summary(), train_fraction=1.0)
    assert a["n_points"] == 4
    # cnn1d has both larger gap and larger sdg effect than minirocket
    assert a["per_model"]["cnn1d"]["mean_gap"] > a["per_model"]["minirocket"]["mean_gap"]
    assert a["per_model"]["cnn1d"]["mean_sdg_effect"] > a["per_model"]["minirocket"]["mean_sdg_effect"]


def test_analyze_gap_handles_missing_train_accuracy():
    rows = [{"dataset": "D1", "model": "cnn1d", "augmentation": "none",
             "status": "completed", "accuracy": 0.8}]  # no train_accuracy
    a = analyze_gap(rows, _summary(), train_fraction=1.0)
    assert a["n_points"] == 0
