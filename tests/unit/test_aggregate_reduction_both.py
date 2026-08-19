"""Unit tests for the both-metrics reduction helpers (issue #21 DS-2).

Covers the registered target rule and the accuracy/macro_f1 both-metric wrapper
used to build results.json['reduction_pamap2'] (src/signal_aug/reporting/aggregate.py).
"""

from __future__ import annotations

import yaml

from signal_aug.reporting import aggregate


def test_rule_target_floors_to_005():
    # floor_0.05(mean - 0.05)
    assert aggregate._rule_target(0.7656) == 0.70   # 0.7156 -> 0.70
    assert aggregate._rule_target(0.83) == 0.75     # 0.78  -> 0.75
    assert aggregate._rule_target(0.905) == 0.85    # 0.855 -> 0.85


def test_fullpool_none_mean_uses_max_subject_count():
    rows = [
        {"augmentation": "none", "subject_count": 2, "macro_f1": 0.4, "accuracy": 0.5},
        {"augmentation": "none", "subject_count": 5, "macro_f1": 0.70, "accuracy": 0.80},
        {"augmentation": "none", "subject_count": 5, "macro_f1": 0.80, "accuracy": 0.82},
        {"augmentation": "mixup", "subject_count": 5, "macro_f1": 0.99, "accuracy": 0.99},
    ]
    mean, n_max, n_rep = aggregate._fullpool_none_mean(rows, "macro_f1")
    assert n_max == 5 and n_rep == 2
    assert abs(mean - 0.75) < 1e-9  # (0.70 + 0.80)/2, mixup ignored


def _rows():
    rows = []
    # none rises 0.5 -> 0.8 across N; mixup uniformly higher so it reaches target sooner
    none_curve = {2: 0.50, 3: 0.62, 4: 0.72, 5: 0.78}
    mixup_curve = {2: 0.64, 3: 0.74, 4: 0.80, 5: 0.84}
    for n in (2, 3, 4, 5):
        for r in range(3):
            rows.append({"augmentation": "none", "subject_count": n, "status": "completed",
                         "macro_f1": none_curve[n], "accuracy": none_curve[n] + 0.02})
            rows.append({"augmentation": "mixup", "subject_count": n, "status": "completed",
                         "macro_f1": mixup_curve[n], "accuracy": mixup_curve[n] + 0.02})
    return rows


def test_reduction_both_metrics_uses_config_primary_and_rule_secondary(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "_subject_metric_rows", lambda *a, **k: _rows())
    cfg = tmp_path / "pamap2_subject_count.yaml"
    cfg.write_text(yaml.safe_dump({
        "dataset": "PAMAP2", "target_metric": "macro_f1", "target_value": 0.70,
        "pre_registered": True,
    }))
    out = aggregate._reduction_both_metrics(tmp_path, cfg, prefix="pamap2_subject_count_")
    assert out["dataset"] == "PAMAP2" and out["primary_metric"] == "macro_f1"
    assert set(out["by_metric"]) == {"macro_f1", "accuracy"}
    # primary metric target comes from the config
    mf = out["by_metric"]["macro_f1"]
    assert mf["target_value"] == 0.70
    assert mf["target_source"] == "pre-registered (config target_value)"
    # secondary metric target is derived by the rule from the full-pool none mean
    ac = out["by_metric"]["accuracy"]
    assert ac["target_source"] == "registered rule from full-pool none baseline"
    # full-pool none accuracy mean = 0.80 -> rule floor_0.05(0.75) = 0.75
    assert ac["fullpool_none_mean"] == 0.80 and ac["target_value"] == 0.75
    # both metrics produce a none baseline N* and a reduction table
    assert mf["n_star_none"] is not None and mf["methods"]
