"""Build the static HTML report (report/dist/index.html).

Pipeline: results.json + config + artifacts -> Jinja2 template -> Tailwind CSS
build -> self-contained offline HTML. Report content is fully data-driven;
nothing is hand-typed into the HTML (spec sections 3.10, 9).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

# Single source of augmentation colors. Any augmentation not listed falls back
# to _FALLBACK_COLOR, and charts enumerate augmentations present in the data
# (not this dict), so a new method never silently vanishes from a figure.
AUG_COLORS = {
    "none": "#64748b",
    "oversample": "#0ea5e9",
    "jitter": "#22c55e",
    "scaling": "#eab308",
    "mixup": "#ec4899",
    "dtw": "#8b5cf6",
    "smote": "#f97316",
    "label_shuffle": "#94a3b8",
}
_FALLBACK_COLOR = "#334155"


def aug_color(aug: str) -> str:
    return AUG_COLORS.get(aug, _FALLBACK_COLOR)


def learning_curve_svg(dataset: str, model: str, curves: dict, width: int = 320, height: int = 200) -> str:
    """Inline SVG line chart of accuracy vs train_fraction, one line per
    augmentation. Self-contained (no JS/external refs) for offline viewing."""
    pad_l, pad_r, pad_t, pad_b = 40, 8, 12, 28
    prefix = f"{dataset}|{model}|"
    # enumerate augmentations present in the data (not AUG_COLORS) so a method
    # without a preassigned color still appears
    series = {
        key[len(prefix):]: pts
        for key, pts in sorted(curves.items())
        if key.startswith(prefix)
    }
    if not series:
        return ""
    all_pts = [p for pts in series.values() for p in pts]
    ys = [p["accuracy_mean"] for p in all_pts]
    y_min, y_max = min(ys), max(ys)
    if y_max - y_min < 0.02:
        y_min, y_max = y_min - 0.01, y_max + 0.01
    fracs = sorted({p["train_fraction"] for p in all_pts})
    x_min, x_max = min(fracs), max(fracs)

    def sx(f: float) -> float:
        return pad_l + (f - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(a: float) -> float:
        return pad_t + (1 - (a - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="{dataset} {model} learning curve">']
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    for a in (y_min, y_max):
        parts.append(f'<text x="{pad_l-4}" y="{sy(a)+3:.1f}" text-anchor="end" font-size="9" fill="#64748b">{a:.3f}</text>')
    for f in fracs:
        parts.append(f'<text x="{sx(f):.1f}" y="{height-pad_b+12}" text-anchor="middle" font-size="9" fill="#64748b">{int(f*100)}%</text>')
    # lines
    for aug, pts in series.items():
        color = aug_color(aug)
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['train_fraction']):.1f},{sy(p['accuracy_mean']):.1f}"
                     for i, p in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for p in pts:
            parts.append(f'<circle cx="{sx(p["train_fraction"]):.1f}" cy="{sy(p["accuracy_mean"]):.1f}" r="2" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def subject_curve_svg(curves_by_aug: dict, target: float | None, width: int = 560, height: int = 300) -> str:
    """Wider learning-curve chart for the subject-count study: accuracy vs
    number of subjects, one line per augmentation, with the pre-registered
    target as a horizontal reference line. Self-contained SVG."""
    pad_l, pad_r, pad_t, pad_b = 46, 90, 14, 34
    all_pts = [p for pts in curves_by_aug.values() for p in pts]
    if not all_pts:
        return ""
    ys = [p["accuracy_mean"] for p in all_pts] + ([target] if target else [])
    y_min, y_max = min(ys), max(ys)
    y_min, y_max = y_min - 0.01, y_max + 0.01
    xs = sorted({p["train_fraction"] for p in all_pts})
    x_min, x_max = min(xs), max(xs)

    def sx(x):
        return pad_l + (x - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(a):
        return pad_t + (1 - (a - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" aria-label="subject-count learning curve">']
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    for a in (y_min + 0.01, (y_min + y_max) / 2, y_max - 0.01):
        parts.append(f'<text x="{pad_l-5}" y="{sy(a)+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{a:.2f}</text>')
        parts.append(f'<line x1="{pad_l}" y1="{sy(a):.1f}" x2="{width-pad_r}" y2="{sy(a):.1f}" stroke="#f1f5f9"/>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{height-pad_b+14}" text-anchor="middle" font-size="10" fill="#64748b">{int(x)}</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-4}" text-anchor="middle" font-size="10" fill="#475569">被験者数</text>')
    if target:
        parts.append(f'<line x1="{pad_l}" y1="{sy(target):.1f}" x2="{width-pad_r}" y2="{sy(target):.1f}" stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(target)+3:.1f}" font-size="10" fill="#ef4444">目標 {target}</text>')
    for aug, pts in curves_by_aug.items():
        color = aug_color(aug)
        pts = sorted(pts, key=lambda p: p["train_fraction"])
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['train_fraction']):.1f},{sy(p['accuracy_mean']):.1f}"
                     for i, p in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{2 if aug == "none" else 1.5}" '
                     f'{"" if aug == "none" else "opacity=0.85"}/>')
        for p in pts:
            parts.append(f'<circle cx="{sx(p["train_fraction"]):.1f}" cy="{sy(p["accuracy_mean"]):.1f}" r="2.5" fill="{color}"/>')
        last = pts[-1]
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(last["accuracy_mean"])+3:.1f}" font-size="9" fill="{color}">{aug}</text>')
    parts.append("</svg>")
    return "".join(parts)


def cross_reduction_svg(cross: dict, metric: str, width: int = 580, height: int = 320) -> str:
    """Scatter of reduction rate (y) vs pool subject count (x) across the three
    subject datasets, one dot per augmentation, for a single metric. The none
    baseline is the y=0 line. Left-censored datasets (N*(none) at the grid floor)
    are NOT given a false reduction value; instead a shaded band marks them as
    "推定不能". Self-contained SVG (no JS/external refs). Data-driven from
    results.json['reduction_cross']; nothing about the values is hand-typed."""
    datasets = [d for d in cross.get("datasets", []) if metric in d.get("by_metric", {})]
    if not datasets:
        return ""
    pad_l, pad_r, pad_t, pad_b = 46, 96, 34, 40
    # y-range from estimable reductions only (skip none and censored/null)
    yvals = [0.0]
    for d in datasets:
        b = d["by_metric"][metric]
        if b.get("left_censored"):
            continue
        for m in b["methods"]:
            if m["augmentation"] != "none" and m.get("reduction_rate") is not None:
                yvals.append(m["reduction_rate"])
    y_min, y_max = min(yvals), max(yvals)
    span = max(y_max - y_min, 0.1)
    y_min, y_max = y_min - 0.05 * span, y_max + 0.05 * span
    pools = [d["pool_max"] for d in datasets]
    x_min, x_max = min(pools), max(pools)

    def sx(x):
        return pad_l + (x - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="cross-dataset reduction vs pool size ({metric})">']
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    # y ticks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y_min + frac * (y_max - y_min)
        parts.append(f'<text x="{pad_l-5}" y="{sy(v)+3:.1f}" text-anchor="end" font-size="9" fill="#64748b">{v*100:+.0f}%</text>')
    # zero line = none baseline
    parts.append(f'<line x1="{pad_l}" y1="{sy(0):.1f}" x2="{width-pad_r}" y2="{sy(0):.1f}" '
                 f'stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{width-pad_r+4}" y="{sy(0)+3:.1f}" font-size="9" fill="#ef4444">none 基準</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-6}" text-anchor="middle" font-size="10" fill="#475569">母集団(pool)被験者数</text>')
    parts.append(f'<text x="14" y="{(pad_t+height-pad_b)/2:.0f}" text-anchor="middle" font-size="10" fill="#475569" '
                 f'transform="rotate(-90 14 {(pad_t+height-pad_b)/2:.0f})">削減率 = 1 − N*(aug)/N*(none)</text>')
    # per-dataset x tick + points
    order = ["oversample", "scaling", "mixup", "dtw", "smote", "label_shuffle"]
    for d in datasets:
        b = d["by_metric"][metric]
        x0 = sx(d["pool_max"])
        parts.append(f'<text x="{x0:.1f}" y="{height-pad_b+14}" text-anchor="middle" font-size="10" fill="#334155">{d["dataset"]}</text>')
        parts.append(f'<text x="{x0:.1f}" y="{height-pad_b+25}" text-anchor="middle" font-size="8" fill="#94a3b8">pool {d["pool_max"]}</text>')
        if b.get("left_censored"):
            # mark as not-estimable rather than plotting a false reduction
            parts.append(f'<rect x="{x0-13:.1f}" y="{pad_t}" width="26" height="{height-pad_t-pad_b}" '
                         f'fill="#94a3b8" opacity="0.12"/>')
            parts.append(f'<text x="{x0:.1f}" y="{(pad_t+height-pad_b)/2:.0f}" text-anchor="middle" font-size="9" '
                         f'fill="#64748b" transform="rotate(-90 {x0:.1f} {(pad_t+height-pad_b)/2:.0f})">推定不能(左側打ち切り)</text>')
            continue
        methods = {m["augmentation"]: m for m in b["methods"]}
        present = [a for a in order if a in methods]
        n = len(present)
        for i, aug in enumerate(present):
            m = methods[aug]
            if m.get("reduction_rate") is None:
                continue
            jitter = (i - (n - 1) / 2) * 5
            cx, cy = x0 + jitter, sy(m["reduction_rate"])
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="{aug_color(aug)}" '
                         f'stroke="#fff" stroke-width="0.6"/>')
    # legend (methods)
    lx = pad_l
    for aug in order:
        parts.append(f'<circle cx="{lx+4:.0f}" cy="{pad_t-16}" r="3.2" fill="{aug_color(aug)}"/>')
        parts.append(f'<text x="{lx+10:.0f}" y="{pad_t-12}" font-size="8" fill="#475569">{aug}</text>')
        lx += 18 + len(aug) * 5.2
    parts.append("</svg>")
    return "".join(parts)


def wesad_curve_svg(methods: list, target: float | None, width: int = 560, height: int = 300) -> str:
    """Subject-count learning curve for WESAD (issue #21 DS-3, non-HAR physio).
    Plots the chosen metric (mean) vs subject count, one line per augmentation.
    The real-data baseline ``none`` and the negative control ``label_shuffle``
    are drawn boldly (the DS-3 story is that they overlap = near-chance), other
    methods are faint context. The pre-registered target is a dashed reference.
    Data-driven from results.json['reduction_wesad']; no values are hand-typed."""
    curves = {m["augmentation"]: sorted(m.get("curve", []), key=lambda p: p["subject_count"])
              for m in methods if m.get("curve")}
    if not curves:
        return ""
    pad_l, pad_r, pad_t, pad_b = 46, 96, 16, 34
    all_pts = [p for pts in curves.values() for p in pts]
    ys = [p["mean"] for p in all_pts] + ([target] if target is not None else [])
    y_min, y_max = min(ys) - 0.02, max(ys) + 0.02
    xs = sorted({p["subject_count"] for p in all_pts})
    x_min, x_max = min(xs), max(xs)

    def sx(x):
        return pad_l + (x - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    def sy(a):
        return pad_t + (1 - (a - y_min) / (y_max - y_min or 1)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="WESAD subject-count learning curve (physiological, 3-class)">']
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#cbd5e1"/>')
    for a in (y_min + 0.02, (y_min + y_max) / 2, y_max - 0.02):
        parts.append(f'<text x="{pad_l-5}" y="{sy(a)+3:.1f}" text-anchor="end" font-size="10" fill="#64748b">{a:.2f}</text>')
        parts.append(f'<line x1="{pad_l}" y1="{sy(a):.1f}" x2="{width-pad_r}" y2="{sy(a):.1f}" stroke="#f1f5f9"/>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{height-pad_b+14}" text-anchor="middle" font-size="10" fill="#64748b">{int(x)}</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-4}" text-anchor="middle" font-size="10" fill="#475569">被験者数</text>')
    if target is not None:
        parts.append(f'<line x1="{pad_l}" y1="{sy(target):.1f}" x2="{width-pad_r}" y2="{sy(target):.1f}" '
                     f'stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(target)+3:.1f}" font-size="10" fill="#ef4444">目標 {target:.2f}</text>')
    highlight = {"none", "label_shuffle"}
    # faint context methods first, then highlighted lines on top
    for aug in sorted(curves, key=lambda a: a in highlight):
        pts = curves[aug]
        color = aug_color(aug)
        strong = aug in highlight
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['subject_count']):.1f},{sy(p['mean']):.1f}"
                     for i, p in enumerate(pts))
        dash = ' stroke-dasharray="5 3"' if aug == "label_shuffle" else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" '
                     f'stroke-width="{2.2 if strong else 1}"{dash} opacity="{1 if strong else 0.4}"/>')
        if strong:
            for p in pts:
                parts.append(f'<circle cx="{sx(p["subject_count"]):.1f}" cy="{sy(p["mean"]):.1f}" r="2.5" fill="{color}"/>')
        last = pts[-1]
        parts.append(f'<text x="{width-pad_r+4}" y="{sy(last["mean"])+3:.1f}" font-size="9" '
                     f'fill="{color}" opacity="{1 if strong else 0.55}">{aug}</text>')
    parts.append("</svg>")
    return "".join(parts)


def synthesis_reduction_svg(synthesis: dict, width: int = 640) -> str:
    """Single integrated view (issue #21 §6.7) of subject-count reduction across
    the four subject datasets. A forest plot on the *reduction-rate* axis: for
    the two measurable datasets (UCI HAR / WISDM, each under its own
    pre-registered primary target) every augmentation is a point with a
    horizontal CI whisker; the ``none`` baseline is x=0 with a shaded band for
    its own N* uncertainty. Datasets whose reduction is not estimable (PAMAP2 =
    left-censored, WESAD = near-chance) are NOT given a false reduction value —
    they get an explicit "推定不能" annotation row. Data-driven from the
    per-dataset reduction blocks in results.json; no values are hand-typed."""
    rows = synthesis.get("rows", [])
    measurable = [r for r in rows if r.get("measurable")]
    censored = [r for r in rows if not r.get("measurable")]
    if not measurable:
        return ""
    # x extent from estimable reductions, method CIs and none bands
    xs = [0.0]
    for r in measurable:
        xs += [v for v in r.get("none_band", []) if v is not None]
        for m in r["methods"]:
            xs += [v for v in m.get("red_ci", []) if v is not None]
            if m.get("reduction_rate") is not None:
                xs.append(m["reduction_rate"])
    x_min, x_max = min(xs), max(xs)
    span = max(x_max - x_min, 0.1)
    x_min, x_max = x_min - 0.05 * span, x_max + 0.05 * span

    pad_l, pad_r, pad_t = 150, 20, 56
    row_h, header_h, note_h = 19, 22, 26
    # total height
    h = pad_t
    for r in measurable:
        h += header_h + len(r["methods"]) * row_h
    for _ in censored:
        h += header_h + note_h
    h += 10
    height = int(h)

    def sx(v: float) -> float:
        return pad_l + (v - x_min) / (x_max - x_min or 1) * (width - pad_l - pad_r)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="cross-dataset subject-count reduction synthesis (forest plot)">']
    plot_bottom = height - 6
    # x grid + ticks (reduction %)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = x_min + frac * (x_max - x_min)
        parts.append(f'<line x1="{sx(v):.1f}" y1="{pad_t-6}" x2="{sx(v):.1f}" y2="{plot_bottom}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{sx(v):.1f}" y="{pad_t-10}" text-anchor="middle" font-size="9" fill="#64748b">{v*100:+.0f}%</text>')
    # zero line = none baseline
    parts.append(f'<line x1="{sx(0):.1f}" y1="{pad_t-6}" x2="{sx(0):.1f}" y2="{plot_bottom}" '
                 f'stroke="#ef4444" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{sx(0):.1f}" y="14" text-anchor="middle" font-size="9" fill="#ef4444">none 基準(削減 0)</text>')
    parts.append(f'<text x="8" y="28" font-size="9" fill="#475569">← 実被験者が増(悪化)</text>')
    parts.append(f'<text x="{width-pad_r}" y="28" text-anchor="end" font-size="9" fill="#475569">実被験者が減(削減) →</text>')
    parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="28" text-anchor="middle" font-size="9" fill="#94a3b8">削減率 = 1 − N*(aug)/N*(none)</text>')

    y = pad_t
    for r in measurable:
        # group header
        parts.append(f'<text x="8" y="{y+13:.0f}" font-size="10" font-weight="bold" fill="#0f172a">{r["dataset"]}</text>')
        parts.append(f'<text x="8" y="{y+13:.0f}" font-size="10" fill="#0f172a" text-anchor="start" dx="{len(r["dataset"])*6.6:.0f}">'
                     f'({r["target_metric"]} {r["target_value"]:.2f}, pool {r["pool_max"]})</text>')
        gy0 = y + header_h - 4
        gy1 = gy0 + len(r["methods"]) * row_h
        # none uncertainty band
        nb = r.get("none_band") or []
        if len(nb) == 2 and None not in nb:
            bx0, bx1 = sx(min(nb)), sx(max(nb))
            parts.append(f'<rect x="{bx0:.1f}" y="{gy0:.1f}" width="{max(bx1-bx0,1):.1f}" height="{gy1-gy0:.1f}" '
                         f'fill="#64748b" opacity="0.08"/>')
        y += header_h
        for m in r["methods"]:
            cy = y + row_h / 2
            color = aug_color(m["augmentation"])
            is_ctrl = m.get("is_control")
            label = m["augmentation"] + ("(対照)" if is_ctrl else "")
            parts.append(f'<text x="{pad_l-8}" y="{cy+3:.1f}" text-anchor="end" font-size="9" '
                         f'fill="{"#94a3b8" if is_ctrl else "#334155"}">{label}</text>')
            ci = m.get("red_ci") or []
            if len(ci) == 2 and None not in ci:
                dash = ' stroke-dasharray="3 2"' if is_ctrl else ""
                parts.append(f'<line x1="{sx(ci[0]):.1f}" y1="{cy:.1f}" x2="{sx(ci[1]):.1f}" y2="{cy:.1f}" '
                             f'stroke="{color}" stroke-width="1.4" opacity="0.7"{dash}/>')
                for e in ci:
                    parts.append(f'<line x1="{sx(e):.1f}" y1="{cy-3:.1f}" x2="{sx(e):.1f}" y2="{cy+3:.1f}" stroke="{color}" stroke-width="1"/>')
            if m.get("reduction_rate") is not None:
                parts.append(f'<circle cx="{sx(m["reduction_rate"]):.1f}" cy="{cy:.1f}" r="3.1" fill="{color}" stroke="#fff" stroke-width="0.6"/>')
            y += row_h

    for r in censored:
        parts.append(f'<text x="8" y="{y+13:.0f}" font-size="10" font-weight="bold" fill="#0f172a">{r["dataset"]}</text>')
        parts.append(f'<text x="8" y="{y+13:.0f}" font-size="10" fill="#0f172a" dx="{len(r["dataset"])*6.6:.0f}">'
                     f'({r["target_metric"]} {r["target_value"]:.2f}, pool {r["pool_max"]})</text>')
        y += header_h
        cy = y + note_h / 2
        parts.append(f'<rect x="{pad_l}" y="{y:.1f}" width="{width-pad_l-pad_r}" height="{note_h-4:.1f}" fill="#94a3b8" opacity="0.10"/>')
        parts.append(f'<text x="{(pad_l+width-pad_r)/2:.0f}" y="{cy+3:.1f}" text-anchor="middle" font-size="9.5" fill="#64748b">'
                     f'削減率は推定不能({r["censor_reason"]})</text>')
        y += note_h

    parts.append("</svg>")
    return "".join(parts)


# Palette for per-class waveform traces in the dataset tab (issue #29). Classes
# are enumerated from the data, so a dataset with more classes than colors wraps
# around rather than losing a trace.
CLASS_COLORS = ["#1d4ed8", "#dc2626", "#059669", "#d97706", "#7c3aed",
                "#0891b2", "#be185d", "#65a30d", "#475569", "#c026d3",
                "#0284c7", "#ea580c"]


def class_color(i: int) -> str:
    return CLASS_COLORS[i % len(CLASS_COLORS)]


def _trace_path(values: list, sx, sy) -> str:
    return " ".join(
        f"{'M' if i == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values)
    )


def waveform_svg(traces: list, width: int = 520, row_height: int = 54) -> str:
    """Small-multiples plot of one representative waveform per class (issue #29).

    Each class gets its own row on a shared y-scale, so the reader sees both the
    shape of a single sample and how the classes differ. Values come from
    report/assets/data/dataset_samples.json (built from the training split only)."""
    if not traces:
        return ""
    pad_l, pad_r, pad_t, pad_b = 116, 8, 6, 16
    height = pad_t + pad_b + row_height * len(traces)
    all_values = [v for t in traces for v in t["values"]]
    y_min, y_max = min(all_values), max(all_values)
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    n_points = max(len(t["values"]) for t in traces)

    def sx(i: float) -> float:
        return pad_l + i / max(n_points - 1, 1) * (width - pad_l - pad_r)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="クラス別の代表波形">']
    for row, t in enumerate(traces):
        top = pad_t + row * row_height
        color = class_color(t["class_index"])

        def sy(v: float, _top=top) -> float:
            return _top + 6 + (1 - (v - y_min) / (y_max - y_min)) * (row_height - 16)

        parts.append(f'<line x1="{pad_l}" y1="{top + row_height - 5:.1f}" x2="{width - pad_r}" '
                     f'y2="{top + row_height - 5:.1f}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{top + row_height / 2:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#475569">{t["class_name"]}</text>')
        parts.append(f'<text x="{pad_l - 6}" y="{top + row_height / 2 + 11:.1f}" text-anchor="end" '
                     f'font-size="8.5" fill="#94a3b8">n={t["n_samples"]}</text>')
        parts.append(f'<path d="{_trace_path(t["values"], sx, sy)}" fill="none" stroke="{color}" '
                     f'stroke-width="1.2"/>')
    parts.append(f'<text x="{pad_l}" y="{height - 4}" font-size="9" fill="#94a3b8">0</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end" font-size="9" '
                 f'fill="#94a3b8">系列終端</text>')
    parts.append("</svg>")
    return "".join(parts)


def channel_stack_svg(channel_traces: list, width: int = 520, row_height: int = 40) -> str:
    """All channels of a single window, stacked (issue #29). Multi-channel
    subject datasets otherwise look like a single line in the class plot."""
    if not channel_traces:
        return ""
    pad_l, pad_r, pad_t, pad_b = 116, 8, 6, 6
    height = pad_t + pad_b + row_height * len(channel_traces)
    all_values = [v for t in channel_traces for v in t["values"]]
    y_min, y_max = min(all_values), max(all_values)
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    n_points = max(len(t["values"]) for t in channel_traces)

    def sx(i: float) -> float:
        return pad_l + i / max(n_points - 1, 1) * (width - pad_l - pad_r)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="1窓のチャネル別波形">']
    for row, t in enumerate(channel_traces):
        top = pad_t + row * row_height

        def sy(v: float, _top=top) -> float:
            return _top + 4 + (1 - (v - y_min) / (y_max - y_min)) * (row_height - 10)

        parts.append(f'<text x="{pad_l - 6}" y="{top + row_height / 2 + 3:.1f}" text-anchor="end" '
                     f'font-size="9.5" fill="#475569" font-family="monospace">{t["channel"]}</text>')
        parts.append(f'<path d="{_trace_path(t["values"], sx, sy)}" fill="none" stroke="{class_color(row)}" '
                     f'stroke-width="1.1"/>')
    parts.append("</svg>")
    return "".join(parts)


def class_balance_svg(distribution: list, width: int = 520, bar_height: int = 18) -> str:
    """Horizontal bar chart of the training-split class distribution (issue #29):
    the imbalance that accuracy numbers have to be read against."""
    if not distribution:
        return ""
    pad_l, pad_r, pad_t = 116, 44, 4
    height = pad_t + bar_height * len(distribution) + 4
    total = sum(d["count"] for d in distribution) or 1
    max_count = max(d["count"] for d in distribution) or 1
    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="クラス分布">']
    for i, d in enumerate(distribution):
        y = pad_t + i * bar_height
        w = d["count"] / max_count * (width - pad_l - pad_r)
        parts.append(f'<text x="{pad_l - 6}" y="{y + bar_height / 2 + 3:.1f}" text-anchor="end" '
                     f'font-size="9.5" fill="#475569">{d["class_name"]}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y + 3}" width="{max(w, 1):.1f}" height="{bar_height - 7}" '
                     f'fill="{class_color(i)}" opacity="0.75"/>')
        parts.append(f'<text x="{pad_l + max(w, 1) + 5:.1f}" y="{y + bar_height / 2 + 3:.1f}" '
                     f'font-size="9" fill="#64748b">{d["count"]} ({d["count"] / total * 100:.0f}%)</text>')
    parts.append("</svg>")
    return "".join(parts)


PHASE_NAMES = {
    0: "Phase 0: 基盤構築",
    1: "Phase 1: UCR最小追試",
    2: "Phase 2: UCR横断比較",
    3: "Phase 3: 被験者ID付きデータ選定",
    4: "Phase 4: 被験者数学習曲線",
    5: "Phase 5: 被験者数削減評価",
    6: "Phase 6: 手法改善・別データ検証",
    7: "Phase 7: 統合レポート・研究成果化",
}

REQUIRED_SECTION_IDS = [
    # paper tab (journal-paper structure: abstract, intro, problem setup,
    # proposed framework, related methods, experimental design, results,
    # discussion, limitations, conclusion, references)
    "abstract",
    "introduction",
    # survey explanation + reproduction coverage (issue #31)
    "survey",
    "problem-setup",
    "proposed-method",
    "related-methods",
    "setup",
    "results",
    "learning-curves",
    "subject-reduction",
    "subject-reduction-cross",
    "subject-reduction-wesad",
    "subject-reduction-synthesis",
    "discussion",
    "limitations",
    "conclusion",
    "references",
    # dashboard tab (non-paper: operations, reproducibility, glossary)
    "ops-progress",
    "ops-runs",
    "ops-audit",
    "ops-reproducibility",
    "ops-glossary",
    # dataset tab (issue #29): one section per dataset group, one article per
    # dataset (anchors are generated from the dataset keys)
    "datasets-ucr",
    "datasets-subject",
    # training/evaluation tab (issue #31): per-dataset learning/evaluation
    "eval-protocol",
    # augmentation-method tab (issue #30): one section per method family
    "methods-magnitude",
    "methods-pattern",
    "methods-trivial",
    "methods-control",
    # reproduction & preprocessing tab (reproducibility / portability)
    "repro-steps",
    "repro-preprocessing",
    "repro-judgment",
    "repro-deviations",
]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _markdown_bullets(path: Path) -> list[str]:
    """Extract top-level bullet items from a markdown file."""
    if not path.exists():
        return []
    return [
        line.strip()[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]


def _dataset_entries(root: Path, references_index: dict) -> dict:
    """Assemble the dataset tab (issue #29).

    Three data sources are merged, none of them hand-typed into the HTML:
      * references/dataset_profiles.yaml - prose (what the data is, the task,
        the record schema, caveats). Carries no counts.
      * data/metadata/*.json - structural facts recorded by the loaders
        (channels, window length, sampling rate, split subjects, licence).
      * report/assets/data/dataset_samples.json - counts, class distribution and
        waveform traces extracted from the actual training/pool split.

    A dataset described in the YAML but missing from the generated samples still
    renders (PAMAP2/WESAD are described without waveforms on cost/licence
    grounds); the template shows what is available.
    """
    profiles = _load_yaml(root / "references/dataset_profiles.yaml")
    samples = (_load_json(root / "report/assets/data/dataset_samples.json") or {}).get("datasets", {})
    checksums = _load_json(root / "data/metadata/checksums.json") or {}
    datasets_cfg = _load_yaml(root / "config/datasets.yaml").get("datasets", {})

    groups = profiles.get("groups", {})
    entries: dict[str, list] = {g: [] for g in groups}
    for key, prose in (profiles.get("datasets") or {}).items():
        group = prose.get("group", "ucr")
        sample = samples.get(key, {})
        meta = _load_json(root / f"data/metadata/{key.lower()}.json") or {}
        stats = checksums.get(key, {})

        # class rows: prose descriptions keyed by the class names the data
        # actually carries, so a mismatch shows up as a missing description
        # rather than a silently wrong label
        counts = {d["class_name"]: d["count"] for d in sample.get("class_distribution", [])}
        class_rows = [
            {"name": name, "description": desc, "count": counts.get(str(name))}
            for name, desc in (prose.get("classes") or {}).items()
        ]

        entries.setdefault(group, []).append({
            "key": key,
            "anchor": f"dataset-{key.lower().replace('_', '-')}",
            "display": prose.get("display", key),
            "group": group,
            "role": prose.get("role"),
            "what": prose.get("what"),
            "task": prose.get("task"),
            "schema": prose.get("schema"),
            "caveats": prose.get("caveats"),
            "license": prose.get("license") or meta.get("license"),
            "reference": prose.get("reference"),
            "ref_no": references_index.get(prose.get("reference")),
            "source": meta.get("source"),
            "url": meta.get("url"),
            "note": (datasets_cfg.get(key) or {}).get("note"),
            "classes": class_rows,
            "meta": meta,
            "stats": stats,
            "sample": sample,
            "waveform_svg": waveform_svg(sample.get("traces", [])),
            "channel_svg": channel_stack_svg(sample.get("channel_traces", [])),
            "balance_svg": class_balance_svg(sample.get("class_distribution", [])),
        })
    return {"groups": groups, "entries": entries}


def augmentation_example_svg(source: list, synthetic: list | None, color: str,
                             width: int = 460, height: int = 96) -> str:
    """Before/after trace for one augmentation method (issue #30): the real
    source sample (grey) with the produced synthetic sample overlaid (colored).
    Values come from report/assets/data/augmentation_examples.json, generated by
    running the production augmenters on a training sample."""
    if not source:
        return ""
    pad_l, pad_r, pad_t, pad_b = 8, 8, 8, 8
    values = list(source) + list(synthetic or [])
    y_min, y_max = min(values), max(values)
    if y_max - y_min < 1e-6:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    n_points = max(len(source), len(synthetic or []))

    def sx(i: float) -> float:
        return pad_l + i / max(n_points - 1, 1) * (width - pad_l - pad_r)

    def sy(v: float) -> float:
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * (height - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="w-full h-auto" role="img" '
             f'aria-label="拡張の入出力例">']
    parts.append(f'<path d="{_trace_path(source, sx, sy)}" fill="none" stroke="#cbd5e1" stroke-width="1.6"/>')
    if synthetic:
        parts.append(f'<path d="{_trace_path(synthetic, sx, sy)}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    parts.append("</svg>")
    return "".join(parts)


def _augmentation_entries(root: Path, references_index: dict, results: dict,
                          stats: list, reduction: dict | None) -> dict:
    """Assemble the augmentation-method tab (issue #30).

    Merges, without hand-typing anything into the HTML:
      * references/augmentation_profiles.yaml - prose (idea, procedure, label
        rule, assumption, deviation from the original formulation).
      * config/augmentations.yaml - the parameter VALUES actually used.
      * results.json stats / reduction - the measured effect per model and the
        subject-count reduction on the primary dataset.
      * report/assets/data/augmentation_examples.json - real augmenter output.
    """
    profiles = _load_yaml(root / "references/augmentation_profiles.yaml")
    aug_cfg = _load_yaml(root / "config/augmentations.yaml").get("augmentations", {})
    examples = _load_json(root / "report/assets/data/augmentation_examples.json") or {}
    example_methods = examples.get("methods", {})

    # measured effect per (augmentation, model) on accuracy, Holm-corrected
    effects: dict[str, list] = {}
    for s in stats:
        if s.get("metric") != "accuracy":
            continue
        effects.setdefault(s["augmentation"], []).append({
            "model": s["model"],
            "mean_delta": s.get("mean_delta"),
            "p_value": s.get("p_value"),
            "significant": s.get("significant_holm"),
            "n_pairs": s.get("n_pairs"),
        })
    reduction_by_aug = {
        m["augmentation"]: m for m in (reduction or {}).get("methods", [])
    }

    families = profiles.get("families", {})
    entries: dict[str, list] = {f: [] for f in families}
    for key, prose in (profiles.get("methods") or {}).items():
        family = prose.get("family", "magnitude")
        ex = example_methods.get(key, {})
        entries.setdefault(family, []).append({
            "key": key,
            "anchor": f"aug-{key.replace('_', '-')}",
            "display": prose.get("display", key),
            "family": family,
            "one_liner": prose.get("one_liner"),
            "idea": prose.get("idea"),
            "procedure": prose.get("procedure"),
            "label_rule": prose.get("label"),
            "assumption": prose.get("assumption"),
            "deviation": prose.get("deviation"),
            "role": prose.get("role"),
            "reference": prose.get("reference"),
            "ref_no": references_index.get(prose.get("reference")),
            # parameter meaning (prose) paired with the configured value (config)
            "params": [
                {"name": name, "meaning": meaning,
                 "value": (aug_cfg.get(key, {}).get("params") or {}).get(name)}
                for name, meaning in (prose.get("params") or {}).items()
            ],
            "effects": sorted(effects.get(key, []), key=lambda e: e["model"]),
            "reduction": reduction_by_aug.get(key),
            "example": ex,
            "example_svg": augmentation_example_svg(
                ex.get("source", []), ex.get("synthetic"), aug_color(key)
            ),
            "color": aug_color(key),
        })
    return {
        "families": families,
        "entries": entries,
        "example_meta": {k: v for k, v in examples.items() if k != "methods"},
        "reduction_dataset": (reduction or {}).get("dataset", "UCI HAR"),
        "reduction_target": (reduction or {}).get("target_value"),
    }


def _evaluation_entries(root: Path, results: dict, summary_main: list, curves: dict,
                        dataset_tab: dict) -> dict:
    """Assemble the training/evaluation tab (issue #31).

    The paper tab states the study-wide protocol; this answers the per-dataset
    question the issue raises — "何を学習して、何を評価しているのか" — for every
    dataset separately: the input a model sees, the split it is evaluated on,
    the metric, how many runs backed it, and the resulting numbers.

    Everything is derived from results.json (runs/summary/learning curves) and
    the dataset tab's structural facts; no per-dataset numbers are hand-typed.
    """
    prose = _load_yaml(root / "references/dataset_profiles.yaml")
    profiles = prose.get("datasets") or {}
    entries_by_key = {e["key"]: e for group in dataset_tab["entries"].values() for e in group}

    runs = [r for r in results.get("runs", []) if r.get("status") == "completed"]
    runs_by_dataset: dict[str, list] = {}
    for r in runs:
        if r.get("dataset") == "synthetic":
            continue
        runs_by_dataset.setdefault(r["dataset"], []).append(r)

    rows = []
    for key, run_list in sorted(runs_by_dataset.items()):
        profile = profiles.get(key, {})
        info = entries_by_key.get(key, {})
        sample = info.get("sample", {})
        meta = info.get("meta", {})
        models = sorted({r.get("model") for r in run_list if r.get("model")})
        augs = sorted({r.get("augmentation") for r in run_list if r.get("augmentation")})
        seeds = sorted({r.get("seed") for r in run_list if r.get("seed") is not None})
        fractions = sorted({r.get("train_fraction") for r in run_list
                            if r.get("train_fraction") is not None})
        subject_counts = sorted({r.get("subject_count") for r in run_list
                                 if r.get("subject_count") is not None})

        # full-training-set rows for this dataset, best/worst deltas
        ds_summary = [s for s in summary_main
                      if s["dataset"] == key and s.get("train_fraction", 1.0) == 1.0]
        deltas = [s for s in ds_summary if s.get("delta_vs_none") is not None]
        best = max(deltas, key=lambda s: s["delta_vs_none"], default=None)
        worst = min(deltas, key=lambda s: s["delta_vs_none"], default=None)

        panels = []
        for model in models:
            if len(curves.get(f"{key}|{model}|none", [])) >= 2:
                svg = learning_curve_svg(key, model, curves)
                if svg:
                    panels.append({"model": model, "svg": svg})

        group = profile.get("group", "ucr")
        rows.append({
            "key": key,
            "anchor": f"eval-{key.lower().replace('_', '-')}",
            "display": profile.get("display", key),
            "group": group,
            "task": profile.get("task"),
            "role": profile.get("role"),
            # what the model actually consumes
            "input_shape": {
                "channels": sample.get("n_channels") or (len(meta.get("channels", [])) or None),
                "length": sample.get("length") or meta.get("window_length"),
                "n_classes": sample.get("n_classes") or meta.get("n_classes"),
            },
            "split": (
                "公式の被験者非跨り分割" if group == "subject" and key == "UCI_HAR"
                else "被験者非跨り分割(seed 0 固定・事前登録)" if group == "subject"
                else "UCR 公式の train/test 分割"
            ),
            "eval_unit": "被験者を跨がない held-out 被験者" if group == "subject" else "公式 test split",
            "models": models,
            "augmentations": augs,
            "seeds": seeds,
            "fractions": fractions,
            "subject_counts": subject_counts,
            "n_runs": len(run_list),
            "summary": sorted(ds_summary, key=lambda s: (s["model"], s["augmentation"])),
            "best": best,
            "worst": worst,
            "panels": panels,
        })

    return {
        "groups": prose.get("groups", {}),
        "rows": rows,
        "n_datasets": len(rows),
        "n_runs": sum(r["n_runs"] for r in rows),
    }


def _survey_overview(root: Path, references_index: dict, findings: list) -> dict:
    """Survey-explanation context (issue #31): what Iwana & Uchida (2021) claims,
    which parts this study reproduced, and which it did not touch. Findings are
    linked by id so a claim's evidence stays tied to the generated findings."""
    data = _load_yaml(root / "references/survey_overview.yaml")
    by_id = {f["id"]: f for f in findings}
    claims = []
    for claim in data.get("claims", []):
        claims.append({
            **claim,
            "findings": [by_id[fid] for fid in claim.get("our_evidence", []) if fid in by_id],
        })
    survey = data.get("survey", {})
    return {
        "survey": survey,
        "ref_no": references_index.get(survey.get("reference")),
        "taxonomy": data.get("taxonomy", []),
        "claims": claims,
        "beyond": data.get("beyond_survey", []),
        "n_reproduced": sum(1 for c in claims if c["our_status"] == "reproduced"),
        "n_claims": len(claims),
    }


def gather_context(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root)
    results = _load_json(root / "report/assets/data/results.json") or {
        "runs": [],
        "summary": [],
        "failed_runs": [],
        "audit": None,
    }
    task_queue = _load_yaml(root / "artifacts/task_queue.yaml")
    tasks = task_queue.get("tasks", [])
    current_phase = task_queue.get("current_phase", 0)
    references = _load_json(root / "report/assets/data/references.json") or []

    completed_runs = [r for r in results["runs"] if r["status"] == "completed"]
    reproducibility = {}
    if completed_runs:
        latest = max(completed_runs, key=lambda r: r.get("ended_at") or "")
        reproducibility = {
            "git_commit": latest.get("git_commit", ""),
            "python_version": latest.get("python_version", ""),
            "n_dirty_runs": sum(1 for r in completed_runs if r.get("git_dirty")),
            "n_completed": len(completed_runs),
        }

    # smoke runs on synthetic data are quality-gate checks, not study results
    summary_main = [s for s in results["summary"] if s["dataset"] != "synthetic"]

    baseline = {
        (s["dataset"], s.get("train_fraction", 1.0), s["model"]): s
        for s in summary_main
        if s["augmentation"] == "none"
    }
    for s in summary_main:
        base = baseline.get((s["dataset"], s.get("train_fraction", 1.0), s["model"]))
        s["delta_vs_none"] = (
            round(s["accuracy_mean"] - base["accuracy_mean"], 4) if base and s["augmentation"] != "none" else None
        )
        s["baseline_std"] = base["accuracy_std"] if base else None

    deltas = sorted(
        [s for s in summary_main if s.get("delta_vs_none") is not None],
        key=lambda s: s["delta_vs_none"],
        reverse=True,
    )
    best_improvements = deltas[:3]
    worst_degradations = list(reversed(deltas[-3:])) if deltas else []

    findings_data = _load_json(root / "artifacts/findings.json") or {}
    references_index = {r["key"]: i + 1 for i, r in enumerate(references)}

    # learning-curve charts: one panel per (dataset, model) that has a sweep
    curves = results.get("learning_curves", {})
    curve_keys = sorted({(k.split("|")[0], k.split("|")[1]) for k in curves})
    curve_panels = []
    for dataset, model in curve_keys:
        # only render panels that actually have multiple fractions (a real sweep)
        sample = curves.get(f"{dataset}|{model}|none", [])
        if len(sample) >= 2:
            svg = learning_curve_svg(dataset, model, curves)
            if svg:
                curve_panels.append({"dataset": dataset, "model": model, "svg": svg})

    # subject-count reduction (Phase 5) + its learning-curve chart
    reduction = results.get("reduction")
    reduction_svg = ""
    if reduction and reduction.get("methods"):
        curves_by_aug = {m["augmentation"]: [{"train_fraction": p["subject_count"],
                                              "accuracy_mean": p["mean"]} for p in m["curve"]]
                         for m in reduction["methods"]}
        reduction_svg = subject_curve_svg(curves_by_aug, reduction.get("target_value"))

    # WISDM external-validity reduction (issue #21 DS-1) + its chart
    reduction_wisdm = results.get("reduction_wisdm")
    reduction_wisdm_svg = ""
    if reduction_wisdm and reduction_wisdm.get("methods"):
        curves_w = {m["augmentation"]: [{"train_fraction": p["subject_count"],
                                         "accuracy_mean": p["mean"]} for p in m["curve"]]
                    for m in reduction_wisdm["methods"]}
        reduction_wisdm_svg = subject_curve_svg(curves_w, reduction_wisdm.get("target_value"))

    # WESAD external-validity reduction (issue #21 DS-3, non-HAR physiological
    # signal). Independent "signal-type axis": NOT merged into the HAR cross
    # figure (§6.5) because task/signal-type/class-count are confounded. Its
    # primary-metric (macro-F1) curve highlights none vs label_shuffle overlap.
    reduction_wesad = results.get("reduction_wesad")
    reduction_wesad_svg = ""
    wesad_primary = None
    wesad_primary_block = None
    if reduction_wesad and reduction_wesad.get("by_metric"):
        wesad_primary = reduction_wesad.get("primary_metric", "macro_f1")
        wesad_primary_block = reduction_wesad["by_metric"].get(wesad_primary)
        if wesad_primary_block and wesad_primary_block.get("methods"):
            reduction_wesad_svg = wesad_curve_svg(
                wesad_primary_block["methods"], wesad_primary_block.get("target_value")
            )

    # PAMAP2 external-validity reduction (issue #21 DS-2), both metrics
    reduction_pamap2 = results.get("reduction_pamap2")

    # cross-dataset reduction-vs-pool-size figure (issue #21 DS-2), unified rule
    reduction_cross = results.get("reduction_cross")
    reduction_cross_svg = {}
    pamap2_none_n2 = None
    if reduction_cross and reduction_cross.get("datasets"):
        for metric in reduction_cross.get("metrics", ["accuracy", "macro_f1"]):
            svg = cross_reduction_svg(reduction_cross, metric)
            if svg:
                reduction_cross_svg[metric] = svg
        # the left-censoring evidence value: PAMAP2 none macro-F1 already at N=2
        _p2 = next((d for d in reduction_cross["datasets"] if d["dataset"] == "PAMAP2"), None)
        if _p2 and "macro_f1" in _p2["by_metric"]:
            _none = next((m for m in _p2["by_metric"]["macro_f1"]["methods"]
                          if m["augmentation"] == "none"), None)
            if _none:
                pamap2_none_n2 = next((p["mean"] for p in _none["curve"] if p["subject_count"] == 2), None)

    # ---- issue #21 §6.7: four-dataset synthesis of subject-count reduction ----
    # Uses each dataset's OWN pre-registered primary target (the same lens as
    # F-8/F-10/F-12), so the synthesis restates the main conclusions unchanged.
    # Descriptive labels (signal type / class count) are structural facts, not
    # experiment outputs; every number (target, N*, CI, reduction, pool) is
    # data-driven from the reduction blocks above — nothing hand-typed.
    METHOD_ORDER = ["dtw", "mixup", "scaling", "smote", "oversample", "label_shuffle"]
    pool_by_ds = {d["dataset"]: d["pool_max"] for d in (reduction_cross or {}).get("datasets", [])}

    def _measurable_row(block, dataset, signal, n_classes, pool_max, control_relation):
        nn = block.get("n_star_none")
        nn_ci = block.get("n_star_none_ci") or [None, None]
        mmap = {m["augmentation"]: m for m in block.get("methods", [])}
        methods = []
        for a in METHOD_ORDER:
            m = mmap.get(a)
            if not m or m.get("n_star") is None:
                continue
            ci = m.get("n_star_ci") or [None, None]
            red_ci = [None, None]
            if nn and None not in ci:
                red_ci = [round(1 - ci[1] / nn, 4), round(1 - ci[0] / nn, 4)]
            methods.append({
                "augmentation": a,
                "reduction_rate": m.get("reduction_rate"),
                "red_ci": red_ci,
                "n_star": m.get("n_star"),
                "is_control": a == "label_shuffle",
            })
        none_band = [None, None]
        if nn and None not in nn_ci:
            none_band = [round(1 - nn_ci[1] / nn, 4), round(1 - nn_ci[0] / nn, 4)]
        return {
            "dataset": dataset, "signal": signal, "n_classes": n_classes,
            "pool_max": pool_max, "measurable": True,
            "target_metric": block.get("target_metric"),
            "target_value": block.get("target_value"),
            "n_star_none": nn, "n_star_none_ci": nn_ci,
            "methods": methods, "none_band": none_band,
            "conclusion": "帰無", "control_relation": control_relation,
        }

    synthesis = None
    synthesis_svg = ""
    if reduction and reduction_wisdm:
        rows = [
            _measurable_row(reduction, "UCI HAR", "慣性 6ch(加速度+ジャイロ)", 6,
                            pool_by_ds.get("UCI_HAR", 21),
                            "全手法の N* CI が none と重複。悲観的対照 label_shuffle を CI 分離で超える手法なし"),
            _measurable_row(reduction_wisdm, "WISDM", "加速度 3ch", 6,
                            pool_by_ds.get("WISDM", 24),
                            "同上。点推定最良の手法も CI が none と重複、対照越えなし"),
        ]
        p2 = next((d for d in (reduction_cross or {}).get("datasets", []) if d["dataset"] == "PAMAP2"), None)
        if p2:
            pr = p2.get("pre_registered_primary", {})
            rows.append({
                "dataset": "PAMAP2", "signal": "多IMU 9ch(加速度+ジャイロ+磁気)", "n_classes": 12,
                "pool_max": p2.get("pool_max"), "measurable": False,
                "target_metric": pr.get("metric", "macro_f1"), "target_value": pr.get("value", 0.70),
                "conclusion": "推定不能(左側打ち切り)", "censor_reason": "左側打ち切り",
                "control_relation": "none が最小格子 N で既に target 超過。対照も target 未到達で削減を示さず",
            })
        if reduction_wesad and wesad_primary_block:
            rows.append({
                "dataset": "WESAD", "signal": "胸部生理 5ch(ECG/EDA/EMG/RESP/TEMP)", "n_classes": 3,
                "pool_max": wesad_primary_block.get("fullpool_n"), "measurable": False,
                "target_metric": reduction_wesad.get("primary_metric", "macro_f1"),
                "target_value": wesad_primary_block.get("target_value"),
                "conclusion": "推定不能(near-chance / 枠組み非転移)", "censor_reason": "near-chance",
                "control_relation": "label_shuffle ≈ none(モデル未学習)。削減の前提が不成立",
            })
        synthesis = {
            "rows": rows,
            "n_measurable": sum(1 for r in rows if r["measurable"]),
            "n_datasets": len(rows),
        }
        synthesis_svg = synthesis_reduction_svg(synthesis)

    from signal_aug.evaluation.stats import holm_bonferroni

    stats = results.get("stats", [])
    # significance after Holm-Bonferroni step-down over the family of tests
    stats_sorted = holm_bonferroni(stats, alpha=0.05)

    # main table shows full-training-set rows only; the fraction sweep lives in
    # the learning-curve figures, so 858 rows don't flood the table
    summary_full = [s for s in summary_main if s.get("train_fraction", 1.0) == 1.0]

    # headline numbers for the abstract/intro (all derived from data, not typed)
    study_runs = [r for r in results["runs"] if r["status"] == "completed" and r.get("dataset") != "synthetic"]
    dtw_reduction = next(
        (m["reduction_rate"] for m in (reduction or {}).get("methods", []) if m["augmentation"] == "dtw"),
        None,
    )
    # separate distinct UCR datasets (RQ1) from subject-ID datasets (RQ2) so the
    # abstract/intro/§5.1 don't conflate them (UCR count must not include UCI HAR/WISDM)
    _study_datasets = {r["dataset"] for r in study_runs}
    _ucr_names = {n for n, s in _load_yaml(root / "config/datasets.yaml").get("datasets", {}).items()
                  if s.get("source") == "ucr"}
    facts = {
        "n_study_runs": len(study_runs),
        "n_study_datasets": len(_study_datasets),
        "n_ucr_datasets": len(_study_datasets & _ucr_names),
        "n_subject_datasets": len(_study_datasets - _ucr_names - {"synthetic"}),
        "n_significant": sum(1 for s in stats_sorted if s.get("significant_holm")),
        "target_metric": (reduction or {}).get("target_metric"),
        "target_value": (reduction or {}).get("target_value"),
        "n_star_none": (reduction or {}).get("n_star_none"),
        "dtw_reduction_pct": round(dtw_reduction * 100, 1) if dtw_reduction is not None else None,
    }

    dataset_tab = _dataset_entries(root, references_index)

    return {
        "facts": facts,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "results": results,
        "summary": summary_full,
        "best_improvements": best_improvements,
        "worst_degradations": worst_degradations,
        "findings": findings_data.get("findings", []),
        "ref": references_index,
        "curve_panels": curve_panels,
        "stats": stats_sorted,
        "aug_colors": AUG_COLORS,
        "reduction": reduction,
        "reduction_svg": reduction_svg,
        "reduction_wisdm": reduction_wisdm,
        "reduction_wisdm_svg": reduction_wisdm_svg,
        "reduction_pamap2": reduction_pamap2,
        "reduction_wesad": reduction_wesad,
        "reduction_wesad_svg": reduction_wesad_svg,
        "wesad_primary": wesad_primary,
        "wesad_primary_block": wesad_primary_block,
        "reduction_cross": reduction_cross,
        "reduction_cross_svg": reduction_cross_svg,
        "cross_method_order": ["oversample", "scaling", "mixup", "dtw", "smote", "label_shuffle"],
        "pamap2_none_n2": pamap2_none_n2,
        "synthesis": synthesis,
        "synthesis_svg": synthesis_svg,
        "n_runs": len(results["runs"]),
        "n_completed": len(completed_runs),
        "n_failed": len(results["failed_runs"]),
        "tasks": tasks,
        "tasks_done": [t for t in tasks if t.get("status") == "done"],
        "tasks_doing": [t for t in tasks if t.get("status") == "in_progress"],
        "tasks_todo": [t for t in tasks if t.get("status") == "todo"],
        "current_phase": current_phase,
        "current_phase_name": PHASE_NAMES.get(current_phase, f"Phase {current_phase}"),
        "datasets_cfg": _load_yaml(root / "config/datasets.yaml").get("datasets", {}),
        "augmentations_cfg": _load_yaml(root / "config/augmentations.yaml").get("augmentations", {}),
        "models_cfg": _load_yaml(root / "config/models.yaml").get("models", {}),
        "reproduction_targets": _load_yaml(root / "references/reproduction_targets.yaml"),
        # dataset tab (issue #29): prose + loader metadata + real sample traces
        "dataset_tab": dataset_tab,
        # augmentation-method tab (issue #30): prose + configured params +
        # measured effects + real augmenter output
        "method_tab": _augmentation_entries(root, references_index, results, stats_sorted, reduction),
        # training/evaluation tab (issue #31): per-dataset "what is learned and
        # evaluated", with that dataset's own results
        "eval_tab": _evaluation_entries(root, results, summary_main, curves, dataset_tab),
        # survey explanation + reproduction coverage (issue #31)
        "survey_tab": _survey_overview(root, references_index, findings_data.get("findings", [])),
        "limitations": _markdown_bullets(root / "artifacts/limitations.md"),
        "audit": results.get("audit"),
        "reproducibility": reproducibility,
        "references": references,
        # reproduction & preprocessing tab (data-driven from artifacts/*.yaml;
        # nothing hand-typed into the HTML, spec 3.10 / 9)
        "reproduction_steps": _load_yaml(root / "artifacts/reproduction_steps.yaml"),
        "preprocessing_notes": _load_yaml(root / "artifacts/preprocessing_notes.yaml"),
        "judgment_calls": _load_yaml(root / "artifacts/judgment_calls.yaml"),
        "deviations": _markdown_bullets(root / "artifacts/deviations.md"),
    }


def render_report(context: dict, template_dir: str | Path = "report/src", css: str = "") -> str:
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.template.html")
    return template.render(css=css, **context)


def build_css(repo_root: str | Path = ".", rendered_html_path: Path | None = None) -> str:
    """Run the Tailwind CLI over the rendered HTML. Falls back to the last
    built CSS if node_modules is unavailable (keeps CI/network-free builds working)."""
    root = Path(repo_root)
    report_dir = root / "report"
    css_cache = report_dir / "dist/assets/report.css"
    tailwind_bin = report_dir / "node_modules/.bin/tailwindcss"
    if tailwind_bin.exists() and shutil.which("node"):
        out = css_cache
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(tailwind_bin.resolve()),
                "-i", "src/input.css",
                "-o", str(out.relative_to(report_dir)),
                "--content", str(rendered_html_path.relative_to(report_dir)) if rendered_html_path else "dist/index.html",
                "--minify",
            ],
            cwd=report_dir,
            check=True,
            capture_output=True,
        )
    if css_cache.exists():
        return css_cache.read_text(encoding="utf-8")
    return ""  # unstyled but valid HTML


def build_report(repo_root: str | Path = ".") -> Path:
    root = Path(repo_root)
    context = gather_context(root)
    dist = root / "report/dist"
    dist.mkdir(parents=True, exist_ok=True)

    # two-pass: render for Tailwind content scan, then inline the built CSS
    tmp = root / "report/dist/assets/index.tmp.html"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(render_report(context, root / "report/src", css=""), encoding="utf-8")
    css = build_css(root, rendered_html_path=tmp)
    tmp.unlink(missing_ok=True)

    out = dist / "index.html"
    out.write_text(render_report(context, root / "report/src", css=css), encoding="utf-8")
    return out
