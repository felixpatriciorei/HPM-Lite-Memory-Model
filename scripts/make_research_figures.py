"""Build paper-style figures for HPM-Lite memory experiments.

This replaces the earlier quick/baseline plotting script with publication-style
figures that emphasize raw seed points, explicitly labeled variability, and
architecture/task diagnostics.

Default inputs:
  results/processed/learned_writer_2048_seed_sweep.csv
  results/processed/local_2048_seed_sweep.csv

Optional input, used only if present:
  results/processed/learned_writer_512_seed_sweep.csv

Outputs:
  results/figures/paper/*.png
  results/figures/paper/*.svg
  results/figures/paper/*.pdf
  results/figures/paper/paper_results_table.csv
  results/figures/paper/figure_manifest.csv
  results/figures/paper/figure_audit_report.md

Design choices:
  * Raw seed points are shown instead of hiding everything behind bars.
  * Error bars are sample standard deviation (SD), not SEM or CI.
  * Local-model writer/memory fields are ignored as bookkeeping noise.
  * Wall-time is reported with a hardware caveat when HPM and local were run on
    different GPUs.
  * SVG/PDF are saved for vector editing; PNG is saved for README/web preview.

No seaborn required. Only matplotlib + Python standard library are used.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

Row = Dict[str, str]

# Okabe-Ito style colorblind-safe palette.
# Avoids red/green-only encoding and prints reasonably in grayscale.
BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
PURPLE = "#CC79A7"
BLACK = "#111111"
GREY = "#777777"
LIGHT_GREY = "#EAEAEA"
DARK_GREY = "#333333"
HPM_COLOR = BLUE
LOCAL_COLOR = ORANGE


@dataclass
class Summary:
    name: str
    n: int
    exact_mean: float
    exact_sd: float
    ce_mean: float
    ce_sd: float
    params_mean: float
    vram_gb_mean: float
    wall_sec_mean: float
    eval_eps_mean: float


def configure_style() -> None:
    """Set a restrained, journal-like Matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 360,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.constrained_layout.use": True,
    })


def read_csv(path: Path) -> List[Row]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def write_csv(path: Path, rows: List[Row], headers: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        w.writerows(rows)


def fnum(row: Row, key: str, default: float = float("nan")) -> float:
    value = (row.get(key) or "").strip()
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def good(values: Iterable[float]) -> List[float]:
    return [v for v in values if not math.isnan(v)]


def mnum(values: Iterable[float]) -> float:
    vals = good(values)
    return mean(vals) if vals else float("nan")


def sdnum(values: Iterable[float]) -> float:
    vals = good(values)
    return stdev(vals) if len(vals) > 1 else 0.0


def pct(values: Iterable[float]) -> List[float]:
    return [100.0 * v for v in values]


def seed_key(row: Row) -> Tuple[int, str]:
    raw = (row.get("seed") or "").strip()
    try:
        return int(raw), row.get("run_id", "")
    except ValueError:
        # Fallback: parse common suffixes such as _seed3.
        run_id = row.get("run_id", "")
        marker = "seed"
        if marker in run_id:
            suffix = run_id.rsplit(marker, 1)[-1]
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                try:
                    return int(digits), run_id
                except ValueError:
                    pass
        return 10_000, run_id


def sort_rows(rows: List[Row]) -> List[Row]:
    return sorted(rows, key=seed_key)


def matched_rows(rows: List[Row], n: int) -> List[Row]:
    return sort_rows(rows)[:n]


def summarize(name: str, rows: List[Row]) -> Summary:
    exact = [fnum(r, "eval_answer_exact") for r in rows]
    ce = [fnum(r, "eval_answer_ce") for r in rows]
    params = [fnum(r, "parameters") for r in rows]
    vram = [fnum(r, "peak_vram_mb") / 1024.0 for r in rows]
    wall = [fnum(r, "train_wall_time_sec") for r in rows]
    eps = [fnum(r, "eval_examples_per_sec") for r in rows]
    return Summary(
        name=name,
        n=len(rows),
        exact_mean=mnum(exact),
        exact_sd=sdnum(exact),
        ce_mean=mnum(ce),
        ce_sd=sdnum(ce),
        params_mean=mnum(params),
        vram_gb_mean=mnum(vram),
        wall_sec_mean=mnum(wall),
        eval_eps_mean=mnum(eps),
    )


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def clean_axis(ax, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)


def mean_sd_points(ax, groups: Sequence[Tuple[str, Sequence[float], str]], ylabel: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    """Draw raw points + mean ± sample SD. No bars."""
    jitter_pattern = [-0.055, 0.0, 0.055, 0.11, -0.11]
    for i, (label, values, color) in enumerate(groups):
        vals = good(values)
        for j, value in enumerate(vals):
            ax.scatter(i + jitter_pattern[j % len(jitter_pattern)], value, s=34, color=color,
                       edgecolors=BLACK, linewidths=0.45, zorder=3)
        mu = mnum(vals)
        sd = sdnum(vals)
        ax.errorbar([i], [mu], yerr=[sd], fmt="_", markersize=18, capsize=5,
                    color=BLACK, linewidth=1.4, zorder=4)
        ax.text(i, mu + (0.035 * ((ylim[1] - ylim[0]) if ylim else max(vals + [1]))), f"{mu:.3g}",
                ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    clean_axis(ax)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, manifest: List[Row], title: str, description: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest.append({
        "figure": stem,
        "png": str(png).replace("\\", "/"),
        "svg": str(svg).replace("\\", "/"),
        "pdf": str(pdf).replace("\\", "/"),
        "title": title,
        "description": description,
    })


def rounded_box(ax, x, y, w, h, text, fc="white", ec=BLACK, lw=1.0, fontsize=8, weight="normal"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.05",
                           linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, wrap=True)
    return patch


def arrow(ax, x1, y1, x2, y2, text: Optional[str] = None, color=BLACK, rad=0.0, lw=1.0):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                          linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, text, fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))


def fig_01_schematic(out_dir: Path, manifest: List[Row]) -> None:
    title = "HPM-Lite architecture and KV memory task"
    desc = "Three-panel schematic of the model components, episodic write/read pathway, and benchmark format."
    fig = plt.figure(figsize=(7.2, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 0.9, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_axis_off()

    # Panel a: model computation graph.
    panel_label(ax_a, "a")
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 4)
    rounded_box(ax_a, 0.3, 2.1, 1.1, 0.55, "token\n$x_t$", fc="#F7F7F7", weight="bold")
    rounded_box(ax_a, 1.75, 2.1, 1.25, 0.55, "embed", fc="#F7F7F7")
    rounded_box(ax_a, 3.45, 2.85, 1.55, 0.55, "local mixer\n$l_t$", fc="#EAF4FB", ec=HPM_COLOR, weight="bold")
    rounded_box(ax_a, 3.45, 2.0, 1.55, 0.55, "GRU state\n$r_t$", fc="#EAF4FB", ec=HPM_COLOR, weight="bold")
    rounded_box(ax_a, 3.45, 1.15, 1.55, 0.55, "writer gate", fc="#FFF3E8", ec=LOCAL_COLOR, weight="bold")
    rounded_box(ax_a, 5.75, 1.05, 1.55, 0.85, "episodic\nmemory slots", fc="#FFF3E8", ec=LOCAL_COLOR, weight="bold")
    rounded_box(ax_a, 5.75, 2.35, 1.55, 0.55, "retrieve\n$e_t$", fc="#FFF3E8", ec=LOCAL_COLOR, weight="bold")
    rounded_box(ax_a, 8.0, 2.35, 1.55, 0.85, "router\nsoftmax($l,r,e$)", fc="#F7F7F7", weight="bold")
    rounded_box(ax_a, 8.0, 1.25, 1.55, 0.55, "answer head\n$p(y)$", fc="#F7F7F7", weight="bold")
    arrow(ax_a, 1.4, 2.38, 1.75, 2.38)
    arrow(ax_a, 3.0, 2.38, 3.45, 3.12)
    arrow(ax_a, 3.0, 2.32, 3.45, 2.28)
    arrow(ax_a, 4.23, 2.0, 4.23, 1.70, "FACT?", color=LOCAL_COLOR)
    arrow(ax_a, 5.0, 1.43, 5.75, 1.43, "write", color=LOCAL_COLOR)
    arrow(ax_a, 6.52, 1.9, 6.52, 2.35, "read", color=LOCAL_COLOR)
    arrow(ax_a, 5.0, 3.12, 8.0, 2.93, "$l_t$", color=HPM_COLOR, rad=-0.06)
    arrow(ax_a, 5.0, 2.27, 8.0, 2.72, "$r_t$", color=HPM_COLOR, rad=0.05)
    arrow(ax_a, 7.30, 2.63, 8.0, 2.62, "$e_t$", color=LOCAL_COLOR)
    arrow(ax_a, 8.78, 2.35, 8.78, 1.80)
    ax_a.text(0.3, 3.65, "HPM-style route between local context, recurrent state, and episodic retrieval",
              fontsize=9.2, fontweight="bold", ha="left")

    # Panel b: task timeline.
    panel_label(ax_b, "b")
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 3)
    tokens = ["FACT\nk12 v77", "NOISE", "FACT\nk03 v19", "NOISE …", "QUERY\nk03", "ANSWER\nv19"]
    xs = [0.35, 1.95, 3.2, 4.85, 6.75, 8.3]
    widths = [1.1, 0.75, 1.1, 1.1, 1.0, 1.1]
    colors = ["#FFF3E8", "#F7F7F7", "#FFF3E8", "#F7F7F7", "#EAF4FB", "#EAF4FB"]
    for x, w, tok, fc in zip(xs, widths, tokens, colors):
        rounded_box(ax_b, x, 1.35, w, 0.75, tok, fc=fc, fontsize=7.6, weight="bold" if "FACT" in tok or "QUERY" in tok else "normal")
    for i in range(len(xs) - 1):
        arrow(ax_b, xs[i] + widths[i], 1.72, xs[i + 1], 1.72, lw=0.75)
    ax_b.plot([3.75, 6.95], [1.18, 1.18], color=GREY, linewidth=1.0, linestyle="--")
    ax_b.text(5.25, 0.83, "long gap beyond local window", fontsize=7.6, ha="center", color=DARK_GREY)
    ax_b.text(0.35, 2.55, "Synthetic KV benchmark: answer requires remembering a distant key→value fact",
              fontsize=9.2, fontweight="bold", ha="left")

    # Panel c: write/retrieve distinction and baseline.
    panel_label(ax_c, "c")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 3.2)
    rounded_box(ax_c, 0.45, 1.95, 2.35, 0.65, "HPM-Lite learned writer\nsupervised FACT write gate", fc="#FFF3E8", ec=LOCAL_COLOR, weight="bold")
    rounded_box(ax_c, 3.75, 1.95, 2.15, 0.65, "memory lookup\ntop-1 retrieval", fc="#FFF3E8", ec=LOCAL_COLOR, weight="bold")
    rounded_box(ax_c, 6.85, 1.95, 2.35, 0.65, "answer prediction\nfrom routed state", fc="#EAF4FB", ec=HPM_COLOR, weight="bold")
    arrow(ax_c, 2.8, 2.27, 3.75, 2.27, "store facts", color=LOCAL_COLOR)
    arrow(ax_c, 5.9, 2.27, 6.85, 2.27, "retrieve value", color=HPM_COLOR)
    rounded_box(ax_c, 0.75, 0.75, 8.0, 0.55,
                "Local Transformer baseline: fixed 256-token window, no persistent episodic slots",
                fc="#F7F7F7", ec=GREY, fontsize=7.9)
    ax_c.text(0.45, 2.9, "Evaluation separates writer errors, retrieval errors, and final answer errors",
              fontsize=9.2, fontweight="bold", ha="left")

    save_figure(fig, out_dir, "fig_01_model_task_schematic", manifest, title, desc)


def fig_02_main_results(out_dir: Path, manifest: List[Row], hpm_rows: List[Row], local_rows: List[Row]) -> None:
    title = "Main 2048-token KV memory result"
    desc = "Matched-seed comparison of accuracy and cross entropy, plus seed-wise contrast and resource context."
    hpm = sort_rows(hpm_rows)
    local = matched_rows(local_rows, len(hpm))
    hpm_exact = pct([fnum(r, "eval_answer_exact") for r in hpm])
    local_exact = pct([fnum(r, "eval_answer_exact") for r in local])
    hpm_ce = [fnum(r, "eval_answer_ce") for r in hpm]
    local_ce = [fnum(r, "eval_answer_ce") for r in local]

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    mean_sd_points(ax_a, [("HPM-Lite\nlearned", hpm_exact, HPM_COLOR), ("Local\nTransformer", local_exact, LOCAL_COLOR)],
                   ylabel="Exact answer accuracy (%)", ylim=(-5, 105))
    ax_a.set_title("Exact accuracy")
    ax_a.text(0.5, 0.08, f"mean gap = {mnum(hpm_exact) - mnum(local_exact):.1f} pp", transform=ax_a.transAxes,
              ha="center", va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor=LIGHT_GREY, pad=2))

    panel_label(ax_b, "b")
    ymax = max(hpm_ce + local_ce) * 1.18 if hpm_ce + local_ce else 1
    mean_sd_points(ax_b, [("HPM-Lite\nlearned", hpm_ce, HPM_COLOR), ("Local\nTransformer", local_ce, LOCAL_COLOR)],
                   ylabel="Answer CE (lower is better)", ylim=(-0.1, ymax))
    ax_b.set_title("Answer cross entropy")

    panel_label(ax_c, "c")
    label_offsets = [1.4, 0.0, -1.4, -2.8, 2.8]
    for i, (h, l) in enumerate(zip(hpm_exact, local_exact)):
        ax_c.plot([0, 1], [l, h], color=GREY, linewidth=1.0, alpha=0.9)
        ax_c.scatter([0], [l], color=LOCAL_COLOR, edgecolors=BLACK, linewidths=0.45, zorder=3)
        ax_c.scatter([1], [h], color=HPM_COLOR, edgecolors=BLACK, linewidths=0.45, zorder=3)
        ax_c.text(1.04, h + label_offsets[i % len(label_offsets)], f"seed {i}", fontsize=7, va="center")
    ax_c.set_xlim(-0.25, 1.35)
    ax_c.set_ylim(-5, 105)
    ax_c.set_xticks([0, 1], ["Local", "HPM-Lite"])
    ax_c.set_ylabel("Exact accuracy (%)")
    ax_c.set_title("Matched seed contrast")
    clean_axis(ax_c)

    panel_label(ax_d, "d")
    ax_d.axis("off")
    hsum = summarize("HPM-Lite", hpm)
    lsum = summarize("Local", local)
    rows = [
        ("Seeds", f"{hsum.n}", f"{lsum.n}"),
        ("Parameters", f"{hsum.params_mean/1000:.1f}K", f"{lsum.params_mean/1000:.1f}K"),
        ("Peak VRAM", f"{hsum.vram_gb_mean:.2f} GB", f"{lsum.vram_gb_mean:.2f} GB"),
        ("Eval speed", f"{hsum.eval_eps_mean:.1f} ex/s", f"{lsum.eval_eps_mean:.1f} ex/s"),
        ("Mean CE", f"{hsum.ce_mean:.3f}", f"{lsum.ce_mean:.3f}"),
    ]
    ax_d.text(0.0, 1.0, "Context, not a speed claim", transform=ax_d.transAxes, fontsize=9.2, fontweight="bold", va="top")
    y = 0.82
    ax_d.text(0.00, y, "Metric", transform=ax_d.transAxes, fontweight="bold")
    ax_d.text(0.45, y, "HPM", transform=ax_d.transAxes, fontweight="bold")
    ax_d.text(0.72, y, "Local", transform=ax_d.transAxes, fontweight="bold")
    y -= 0.11
    for label, hv, lv in rows:
        ax_d.text(0.00, y, label, transform=ax_d.transAxes)
        ax_d.text(0.45, y, hv, transform=ax_d.transAxes)
        ax_d.text(0.72, y, lv, transform=ax_d.transAxes)
        y -= 0.10
    ax_d.text(0.0, 0.03,
              "Wall-clock and throughput depend on hardware. Use this table to check scale, not to claim GPU speed superiority.",
              transform=ax_d.transAxes, fontsize=7.2, color=DARK_GREY, va="bottom", wrap=True)

    save_figure(fig, out_dir, "fig_02_main_2048_results", manifest, title, desc)


def fig_03_diagnostics(out_dir: Path, manifest: List[Row], hpm_rows: List[Row]) -> None:
    title = "HPM-Lite learned-writer and retrieval diagnostics"
    desc = "Diagnostic panels identify whether remaining errors come from writing, retrieval, or answer decoding."
    hpm = sort_rows(hpm_rows)
    exact = pct([fnum(r, "eval_answer_exact") for r in hpm])
    true_write = pct([fnum(r, "eval_true_fact_written_rate") for r in hpm])
    missed = pct([fnum(r, "eval_missed_fact_rate") for r in hpm])
    false_w = pct([fnum(r, "eval_false_write_rate") for r in hpm])
    margin = [fnum(r, "eval_retrieval_margin") for r in hpm]
    top1 = pct([fnum(r, "eval_retrieval_top1") for r in hpm])

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    offsets = [(0.08, 0.10), (0.08, 0.02), (0.08, -0.18), (0.08, 0.24)]
    for i, (m, e) in enumerate(zip(missed, exact)):
        ax_a.scatter(m, e, color=HPM_COLOR, edgecolors=BLACK, linewidths=0.45, s=42)
        dx, dy = offsets[i % len(offsets)]
        ax_a.text(m + dx, e + dy, f"seed {i}", fontsize=7, va="center")
    ax_a.set_xlabel("Missed fact rate (%)")
    ax_a.set_ylabel("Exact accuracy (%)")
    ax_a.set_title("Remaining errors track writer misses")
    ax_a.set_xlim(-0.2, max(missed + [1.0]) + 1.0)
    ax_a.set_ylim(90, 101.5)
    clean_axis(ax_a)

    panel_label(ax_b, "b")
    mean_sd_points(ax_b,
                   [("True\nwrites", true_write, HPM_COLOR), ("Missed\nfacts", missed, LOCAL_COLOR), ("False\nwrites", false_w, PURPLE)],
                   ylabel="Rate (%)", ylim=(-1, 105))
    ax_b.set_title("Writer behavior")

    panel_label(ax_c, "c")
    mean_sd_points(ax_c, [("Retrieval\nmargin", margin, GREEN)],
                   ylabel="Retrieval margin", ylim=(0, max(margin + [1]) * 1.35))
    ax_c.set_title("Retrieval margin")
    ax_c.text(0.5, 0.10, f"Retrieval top-1 = {mnum(top1):.0f}% for all shown seeds", transform=ax_c.transAxes,
              ha="center", fontsize=7.4, color=DARK_GREY,
              bbox=dict(facecolor="white", edgecolor=LIGHT_GREY, pad=2))

    panel_label(ax_d, "d")
    ax_d.axis("off")
    notes = [
        ("Observation", "Retrieval top-1 is 100% across these HPM runs."),
        ("Error source", "Exact drops occur in the seeds with nonzero missed-fact rates."),
        ("Caution", "n=3 seeds: this is descriptive diagnostics, not a formal causal proof."),
        ("Paper wording", "Use: ‘consistent with writer misses.’\nAvoid: ‘proves the writer is the only failure mode.’"),
    ]
    y = 0.95
    for head, body in notes:
        ax_d.text(0.02, y, head, transform=ax_d.transAxes, fontsize=8.2, fontweight="bold", va="top")
        ax_d.text(0.02, y - 0.055, body, transform=ax_d.transAxes, fontsize=7.6, va="top", wrap=True)
        y -= 0.235

    save_figure(fig, out_dir, "fig_03_writer_retrieval_diagnostics", manifest, title, desc)


def row_step_path(row: Row, repo_root: Path) -> Optional[Path]:
    raw = (row.get("step_log_path") or "").strip()
    if not raw:
        return None
    p = Path(raw.replace("\\", "/"))
    candidates = [p, repo_root / p]
    # If processed CSV points to a copied/moved run folder, try matching by run id.
    run_id = (row.get("run_id") or "").strip()
    if run_id:
        candidates.append(repo_root / "runs" / "memory_model" / run_id / "step_log.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_step_log(path: Path) -> List[Row]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    except Exception:
        return []


def first_key(rows: List[Row], options: Sequence[str]) -> Optional[str]:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for key in options:
        if key in keys:
            return key
    return None


def fig_04_learning_curves(out_dir: Path, manifest: List[Row], hpm_rows: List[Row], repo_root: Path) -> bool:
    title = "HPM-Lite training dynamics at 2048 tokens"
    desc = "Step-log curves for exact accuracy, answer CE, writer recall, and loss."
    curves = []
    for row in sort_rows(hpm_rows):
        path = row_step_path(row, repo_root)
        if path is None:
            continue
        log_rows = read_step_log(path)
        if not log_rows:
            continue
        step_key = first_key(log_rows, ["step"])
        exact_key = first_key(log_rows, ["eval_answer_exact", "eval_exact", "answer_exact"])
        ce_key = first_key(log_rows, ["eval_answer_ce", "eval_ce", "answer_ce"])
        writer_key = first_key(log_rows, ["writer_recall", "eval_true_fact_written_rate", "train_writer_true_fact_written_rate"])
        loss_key = first_key(log_rows, ["loss", "train_loss"])
        if not step_key:
            continue
        curves.append({
            "seed": row.get("seed", "?"),
            "steps": [fnum(r, step_key) for r in log_rows],
            "exact": pct([fnum(r, exact_key) for r in log_rows]) if exact_key else [],
            "ce": [math.log1p(max(0.0, fnum(r, ce_key))) for r in log_rows] if ce_key else [],
            "writer": pct([fnum(r, writer_key) for r in log_rows]) if writer_key else [],
            "loss": [math.log1p(max(0.0, fnum(r, loss_key))) for r in log_rows] if loss_key else [],
        })
    if not curves:
        return False

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    specs = [
        ("a", "exact", "Exact accuracy (%)", "Answer accuracy", (0, 105)),
        ("b", "ce", "log1p(answer CE)", "Answer CE", None),
        ("c", "writer", "Writer recall / true write rate (%)", "Writer learning", (70, 105)),
        ("d", "loss", "log1p(training loss)", "Training loss", None),
    ]
    for ax, (letter, key, ylabel, title_text, ylim) in zip(axes, specs):
        panel_label(ax, letter)
        for curve in curves:
            vals = curve[key]
            if vals:
                ax.plot(curve["steps"], vals, marker="o", label=f"seed {curve['seed']}")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title_text)
        if ylim:
            ax.set_ylim(*ylim)
        clean_axis(ax)
    axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig, out_dir, "fig_04_hpm_training_dynamics", manifest, title, desc)
    return True


def fig_05_supplemental(out_dir: Path, manifest: List[Row], local_rows: List[Row], hpm_512_rows: Optional[List[Row]], hpm_2048_rows: List[Row]) -> None:
    title = "Supplemental seed and sequence-length checks"
    desc = "Extra local seed and optional HPM 512 vs 2048 summary."
    local = sort_rows(local_rows)
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_label(ax_a, "a")
    vals = pct([fnum(r, "eval_answer_exact") for r in local])
    seeds = [str(seed_key(r)[0]) for r in local]
    ax_a.scatter(range(len(vals)), vals, color=LOCAL_COLOR, edgecolors=BLACK, linewidths=0.45, s=38)
    ax_a.plot(range(len(vals)), vals, color=LOCAL_COLOR, alpha=0.5)
    ax_a.set_xticks(range(len(vals)), seeds)
    ax_a.set_xlabel("Local baseline seed")
    ax_a.set_ylabel("Exact accuracy (%)")
    ax_a.set_title("All available local seeds")
    ax_a.set_ylim(-1, max(5, max(vals + [0]) + 2))
    clean_axis(ax_a)

    panel_label(ax_b, "b")
    if hpm_512_rows:
        seqs = [512, 2048]
        means = [mnum(pct([fnum(r, "eval_answer_exact") for r in hpm_512_rows])),
                 mnum(pct([fnum(r, "eval_answer_exact") for r in hpm_2048_rows]))]
        sds = [sdnum(pct([fnum(r, "eval_answer_exact") for r in hpm_512_rows])),
               sdnum(pct([fnum(r, "eval_answer_exact") for r in hpm_2048_rows]))]
        ax_b.errorbar(seqs, means, yerr=sds, marker="o", capsize=5, color=HPM_COLOR)
        ax_b.set_xscale("log", base=2)
        ax_b.set_xticks(seqs, [str(s) for s in seqs])
        ax_b.set_xlabel("Sequence length")
        ax_b.set_ylabel("HPM exact accuracy (%)")
        ax_b.set_ylim(90, 101.5)
        ax_b.set_title("HPM scaling check")
        ax_b.text(0.5, 0.06, "Only plotted when 512 processed sweep exists.", transform=ax_b.transAxes,
                  ha="center", fontsize=7.2, color=DARK_GREY)
        clean_axis(ax_b)
    else:
        ax_b.axis("off")
        ax_b.text(0.02, 0.90, "Optional panel skipped", fontsize=9.2, fontweight="bold", transform=ax_b.transAxes)
        ax_b.text(0.02, 0.74,
                  "Place results/processed/learned_writer_512_seed_sweep.csv in the repo to plot 512→2048 scaling.",
                  fontsize=8, transform=ax_b.transAxes, wrap=True)

    save_figure(fig, out_dir, "fig_05_supplemental_seed_checks", manifest, title, desc)


def write_results_table(out_dir: Path, hpm_rows: List[Row], local_rows: List[Row]) -> Path:
    hpm = sort_rows(hpm_rows)
    local_match = matched_rows(local_rows, len(hpm))
    local_all = sort_rows(local_rows)
    summaries = [
        summarize("HPM-Lite learned writer (2048, matched)", hpm),
        summarize("Local Transformer (2048, matched)", local_match),
        summarize("Local Transformer (2048, all available)", local_all),
    ]
    out = out_dir / "paper_results_table.csv"
    headers = [
        "group", "n", "exact_mean", "exact_sd", "answer_ce_mean", "answer_ce_sd",
        "parameters_mean", "peak_vram_gb_mean", "wall_time_sec_mean", "eval_examples_per_sec_mean"
    ]
    rows: List[Row] = []
    for s in summaries:
        rows.append({
            "group": s.name,
            "n": str(s.n),
            "exact_mean": f"{s.exact_mean:.8f}",
            "exact_sd": f"{s.exact_sd:.8f}",
            "answer_ce_mean": f"{s.ce_mean:.8f}",
            "answer_ce_sd": f"{s.ce_sd:.8f}",
            "parameters_mean": f"{s.params_mean:.2f}",
            "peak_vram_gb_mean": f"{s.vram_gb_mean:.4f}",
            "wall_time_sec_mean": f"{s.wall_sec_mean:.4f}",
            "eval_examples_per_sec_mean": f"{s.eval_eps_mean:.4f}",
        })
    write_csv(out, rows, headers)
    return out


def write_manifest(out_dir: Path, manifest: List[Row]) -> Path:
    out = out_dir / "figure_manifest.csv"
    headers = ["figure", "png", "svg", "pdf", "title", "description"]
    write_csv(out, manifest, headers)
    return out


def write_audit(out_dir: Path, hpm_rows: List[Row], local_rows: List[Row], made_training_curves: bool) -> Path:
    hpm = sort_rows(hpm_rows)
    local_match = matched_rows(local_rows, len(hpm))
    h = summarize("HPM", hpm)
    l = summarize("Local matched", local_match)
    out = out_dir / "figure_audit_report.md"
    lines = [
        "# Figure audit report",
        "",
        "This file is generated by `scripts/make_research_figures.py` so the figures are auditable.",
        "",
        "## Data used",
        f"- HPM-Lite learned-writer rows used for main comparison: `{len(hpm)}`.",
        f"- Local Transformer rows used for matched comparison: `{len(local_match)}`.",
        f"- Extra local rows available: `{max(0, len(local_rows) - len(local_match))}`.",
        "- Local-model writer/memory columns are ignored in figures because the local baseline has no episodic memory module.",
        "",
        "## Main numerical result",
        f"- HPM exact accuracy: `{100*h.exact_mean:.2f}% ± {100*h.exact_sd:.2f}% SD`.",
        f"- Local exact accuracy, matched seeds: `{100*l.exact_mean:.2f}% ± {100*l.exact_sd:.2f}% SD`.",
        f"- HPM answer CE: `{h.ce_mean:.4f} ± {h.ce_sd:.4f} SD`.",
        f"- Local answer CE: `{l.ce_mean:.4f} ± {l.ce_sd:.4f} SD`.",
        "",
        "## Caveats",
        "- Error bars are sample SD, not SEM or confidence intervals.",
        "- With n=3 matched seeds, the plots are descriptive and should not be oversold as a definitive statistical test.",
        "- Wall-clock speed should not be used as a direct architectural claim if HPM and local runs used different hardware.",
        f"- Training-curve figure generated: `{made_training_curves}`.",
        "",
        "## Figure files",
        "- Vector files: `.svg` and `.pdf` are intended for paper/report editing.",
        "- Raster files: `.png` is for README/web preview.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_optional(path: Path) -> Optional[List[Row]]:
    if path.exists():
        return read_csv(path)
    return None


def validate_rows(hpm_rows: List[Row], local_rows: List[Row]) -> None:
    hpm_rows[:] = [r for r in hpm_rows if r.get("model") == "hpm_lite" and r.get("seq_len") == "2048"]
    local_rows[:] = [r for r in local_rows if r.get("model") == "local" and r.get("seq_len") == "2048"]
    if len(hpm_rows) < 3:
        raise ValueError(f"Expected at least 3 HPM 2048 rows, got {len(hpm_rows)}")
    if len(local_rows) < 3:
        raise ValueError(f"Expected at least 3 local 2048 rows, got {len(local_rows)}")
    for name, rows in [("HPM", hpm_rows), ("local", local_rows)]:
        for row in rows:
            if math.isnan(fnum(row, "eval_answer_exact")):
                raise ValueError(f"{name} row missing eval_answer_exact: {row.get('run_id')}")
            if math.isnan(fnum(row, "eval_answer_ce")):
                raise ValueError(f"{name} row missing eval_answer_ce: {row.get('run_id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style HPM-Lite figures.")
    parser.add_argument("--hpm", type=Path, default=Path("results/processed/learned_writer_2048_seed_sweep.csv"))
    parser.add_argument("--local", type=Path, default=Path("results/processed/local_2048_seed_sweep.csv"))
    parser.add_argument("--hpm512", type=Path, default=Path("results/processed/learned_writer_512_seed_sweep.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    configure_style()
    args.out.mkdir(parents=True, exist_ok=True)

    hpm_rows = read_csv(args.hpm)
    local_rows = read_csv(args.local)
    validate_rows(hpm_rows, local_rows)
    hpm512_rows = load_optional(args.hpm512)
    if hpm512_rows:
        hpm512_rows = [r for r in hpm512_rows if r.get("model") == "hpm_lite" and r.get("seq_len") == "512"]

    manifest: List[Row] = []
    table = write_results_table(args.out, hpm_rows, local_rows)
    print(f"wrote {table}")

    fig_01_schematic(args.out, manifest)
    fig_02_main_results(args.out, manifest, hpm_rows, local_rows)
    fig_03_diagnostics(args.out, manifest, hpm_rows)
    made_training_curves = fig_04_learning_curves(args.out, manifest, hpm_rows, args.repo_root)
    fig_05_supplemental(args.out, manifest, local_rows, hpm512_rows, hpm_rows)

    manifest_path = write_manifest(args.out, manifest)
    audit = write_audit(args.out, hpm_rows, local_rows, made_training_curves)
    print(f"wrote {manifest_path}")
    print(f"wrote {audit}")
    print("wrote figures:")
    for item in manifest:
        print(f"- {item['png']}")


if __name__ == "__main__":
    main()
