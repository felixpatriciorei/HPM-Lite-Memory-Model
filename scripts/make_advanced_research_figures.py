"""Generate research-grade HPM-Lite / HPM-Lite v2 statistics and figures.

This script is intentionally conservative:
- It plots raw seed points whenever seed-level data is available.
- It labels n explicitly.
- It reports sample SD, not confidence intervals, unless enough seeds exist.
- It treats v2 fixed 2048 results with n=1 or n=2 as preliminary.

Outputs:
  results/figures/advanced/
  results/processed/advanced_research_stats.csv
  results/figures/advanced/advanced_figure_manifest.csv
  results/figures/advanced/advanced_figure_audit.md
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures" / "advanced"
PROC_DIR = ROOT / "results" / "processed"
RAW_DIR = ROOT / "results" / "raw"

FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

BASE_HEADERS = [
    "run_id", "model", "task", "write_mode", "seq_len", "window", "seed", "batch_size", "steps",
    "d_model", "layers", "heads", "parameters", "device", "eval_answer_exact", "eval_answer_ce",
    "eval_retrieval_top1", "eval_retrieval_topk", "eval_true_fact_written_rate", "eval_false_write_rate",
    "eval_missed_fact_rate", "eval_avg_written_slots", "eval_retrieval_margin", "train_wall_time_sec",
    "examples_per_sec_recent", "eval_examples_per_sec", "peak_vram_mb", "run_dir", "step_log_path",
]

DATASETS = [
    ("v1 HPM 512", PROC_DIR / "learned_writer_512_seed_sweep.csv"),
    ("v1 HPM 2048", PROC_DIR / "learned_writer_2048_seed_sweep.csv"),
    ("v1 Local 2048", PROC_DIR / "local_2048_seed_sweep.csv"),
    ("v2 HPM 512", PROC_DIR / "hpm_v2_512_seed_sweep.csv"),
    ("v2 HPM 2048 tf600", PROC_DIR / "hpm_v2_2048_tf600_lw03_seed_sweep.csv"),
]

# Optional raw files that may exist but should not be treated as the headline if processed exists.
OPTIONAL_RAW = [
    ("v2 HPM 2048 tf200 seed0", RAW_DIR / "hpm_v2_2048_seed0.csv"),
    ("v2 HPM 2048 tf600 seed0", RAW_DIR / "hpm_v2_2048_seed0_tf600_lw03.csv"),
]


def _float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_str(value: object) -> str:
    f = _float(value)
    if f is None:
        return str(value).strip() if value is not None else ""
    return str(int(f))


def read_csv_rows(path: Path, label: str) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, str]] = []
    for r in rows:
        rr = {k: (v if v is not None else "") for k, v in r.items()}
        rr["dataset_label"] = label
        rr["source_file"] = str(path.relative_to(ROOT))
        out.append(rr)
    return out


def load_data() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for label, path in DATASETS:
        rows.extend(read_csv_rows(path, label))
    return rows


def group_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["dataset_label"], []).append(r)
    return grouped


def sample_sd(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def write_stats(rows: List[Dict[str, str]]) -> Path:
    grouped = group_rows(rows)
    out = PROC_DIR / "advanced_research_stats.csv"
    headers = [
        "dataset", "n", "seq_len", "model", "write_mode", "mean_exact", "sd_exact", "min_exact", "max_exact",
        "mean_ce", "sd_ce", "mean_retrieval_top1", "mean_true_fact_written_rate", "mean_false_write_rate",
        "mean_missed_fact_rate", "mean_wall_time_sec", "mean_peak_vram_mb", "mean_params", "source_files",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for label, rs in grouped.items():
            exact = [v for r in rs if (v := _float(r.get("eval_answer_exact"))) is not None]
            ce = [v for r in rs if (v := _float(r.get("eval_answer_ce"))) is not None]
            ret = [v for r in rs if (v := _float(r.get("eval_retrieval_top1"))) is not None]
            tw = [v for r in rs if (v := _float(r.get("eval_true_fact_written_rate"))) is not None]
            fw = [v for r in rs if (v := _float(r.get("eval_false_write_rate"))) is not None]
            mw = [v for r in rs if (v := _float(r.get("eval_missed_fact_rate"))) is not None]
            wall = [v for r in rs if (v := _float(r.get("train_wall_time_sec"))) is not None]
            vram = [v for r in rs if (v := _float(r.get("peak_vram_mb"))) is not None]
            params = [v for r in rs if (v := _float(r.get("parameters"))) is not None]
            def mean(xs: List[float]) -> str:
                return "" if not xs else f"{float(np.mean(xs)):.6g}"
            def sd(xs: List[float]) -> str:
                return "" if len(xs) < 2 else f"{sample_sd(xs):.6g}"
            w.writerow({
                "dataset": label,
                "n": len(rs),
                "seq_len": ",".join(sorted({_int_str(r.get("seq_len")) for r in rs if r.get("seq_len")})),
                "model": ",".join(sorted({r.get("model", "") for r in rs})),
                "write_mode": ",".join(sorted({r.get("write_mode", "") for r in rs})),
                "mean_exact": mean(exact),
                "sd_exact": sd(exact),
                "min_exact": "" if not exact else f"{min(exact):.6g}",
                "max_exact": "" if not exact else f"{max(exact):.6g}",
                "mean_ce": mean(ce),
                "sd_ce": sd(ce),
                "mean_retrieval_top1": mean(ret),
                "mean_true_fact_written_rate": mean(tw),
                "mean_false_write_rate": mean(fw),
                "mean_missed_fact_rate": mean(mw),
                "mean_wall_time_sec": mean(wall),
                "mean_peak_vram_mb": mean(vram),
                "mean_params": mean(params),
                "source_files": ";".join(sorted({r.get("source_file", "") for r in rs})),
            })
    return out


def savefig(name: str) -> List[Path]:
    paths = []
    for ext in ["png", "svg", "pdf"]:
        p = FIG_DIR / f"{name}.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=240)
        paths.append(p)
    plt.close()
    return paths


def fig_exact_raw_points(rows: List[Dict[str, str]]) -> List[Path]:
    grouped = group_rows(rows)
    labels = [k for k in ["v1 Local 2048", "v1 HPM 2048", "v2 HPM 512", "v2 HPM 2048 tf600"] if k in grouped]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, label in enumerate(labels):
        vals = [v for r in grouped[label] if (v := _float(r.get("eval_answer_exact"))) is not None]
        if not vals:
            continue
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=54, zorder=3)
        mean = float(np.mean(vals))
        ax.plot([i - 0.18, i + 0.18], [mean, mean], linewidth=2.6, zorder=4)
        if len(vals) > 1:
            sd = sample_sd(vals)
            ax.errorbar(i, mean, yerr=sd, capsize=5, linewidth=1.8, zorder=2)
        ax.text(i, min(1.03, mean + 0.055), f"n={len(vals)}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Exact recall by model/configuration with seed-level points")
    ax.set_ylabel("Exact answer accuracy")
    ax.set_ylim(-0.03, 1.08)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.28, "Bars show mean; error bars show sample SD when n≥2. Raw points are individual seeds.", transform=ax.transAxes, fontsize=9)
    return savefig("fig_adv_01_exact_raw_seed_points")


def fig_writer_error_decomposition(rows: List[Dict[str, str]]) -> List[Path]:
    grouped = group_rows(rows)
    labels = [k for k in ["v1 HPM 2048", "v2 HPM 512", "v2 HPM 2048 tf600"] if k in grouped]
    success = []
    missed = []
    false = []
    for label in labels:
        rs = grouped[label]
        success.append(np.mean([_float(r.get("eval_true_fact_written_rate")) or 0 for r in rs]) * 100)
        missed.append(np.mean([_float(r.get("eval_missed_fact_rate")) or 0 for r in rs]) * 100)
        false.append(np.mean([_float(r.get("eval_false_write_rate")) or 0 for r in rs]) * 100)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, success, label="True facts written")
    ax.bar(x, missed, bottom=success, label="Missed facts")
    bottom = np.array(success) + np.array(missed)
    ax.bar(x, false, bottom=bottom, label="False writes")
    ax.set_title("Writer behavior explains most remaining exact-recall errors")
    ax.set_ylabel("Mean rate across seeds (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return savefig("fig_adv_02_writer_error_decomposition")


def fig_writer_vs_exact(rows: List[Dict[str, str]]) -> List[Path]:
    pts = []
    for r in rows:
        tw = _float(r.get("eval_true_fact_written_rate"))
        exact = _float(r.get("eval_answer_exact"))
        ce = _float(r.get("eval_answer_ce")) or 0.0
        if tw is None or exact is None:
            continue
        pts.append((tw, exact, ce, r.get("dataset_label", ""), r.get("seed", "")))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for tw, exact, ce, label, seed in pts:
        ax.scatter(tw, exact, s=52 + min(160, ce * 25), alpha=0.75)
        if label.startswith("v2 HPM 2048"):
            ax.annotate(f"s{seed}", (tw, exact), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Writer success versus exact recall")
    ax.set_xlabel("True fact written rate")
    ax.set_ylabel("Exact answer accuracy")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.20, "Point size increases with answer CE. Seed labels shown for v2 2048 fixed-config points.", transform=ax.transAxes, fontsize=9)
    return savefig("fig_adv_03_writer_success_vs_exact")


def fig_efficiency_frontier(rows: List[Dict[str, str]]) -> List[Path]:
    grouped = group_rows(rows)
    labels = [k for k in grouped.keys()]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for label in labels:
        rs = grouped[label]
        exact = [v for r in rs if (v := _float(r.get("eval_answer_exact"))) is not None]
        wall = [v for r in rs if (v := _float(r.get("train_wall_time_sec"))) is not None]
        vram = [v for r in rs if (v := _float(r.get("peak_vram_mb"))) is not None]
        params = [v for r in rs if (v := _float(r.get("parameters"))) is not None]
        if not exact or not wall:
            continue
        x = float(np.mean(wall))
        y = float(np.mean(exact))
        size = 60 if not vram else 35 + float(np.mean(vram)) / 65
        ax.scatter(x, y, s=size, alpha=0.8)
        ptxt = "" if not params else f", {np.mean(params)/1e6:.2f}M params"
        ax.annotate(f"{label}\nn={len(rs)}{ptxt}", (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_title("Accuracy versus wall-clock cost")
    ax.set_xlabel("Mean train wall time (s)")
    ax.set_ylabel("Mean exact answer accuracy")
    ax.set_ylim(-0.03, 1.06)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.20, "Bubble area roughly scales with peak VRAM. Cross-hardware timing should be treated as approximate.", transform=ax.transAxes, fontsize=9)
    return savefig("fig_adv_04_efficiency_frontier")


def read_step_log(path: Path, label: str) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        step = _float(r.get("step"))
        if step is None:
            continue
        out.append({
            "label": label,
            "step": step,
            "exact": _float(r.get("eval_answer_exact")) or _float(r.get("eval_exact")),
            "ce": _float(r.get("eval_answer_ce")) or _float(r.get("eval_ce")),
            "writer": _float(r.get("writer_recall")) or _float(r.get("eval_true_fact_written_rate")),
            "loss": _float(r.get("loss")) or _float(r.get("train_loss")),
        })
    return out


def collect_step_logs(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    logs: List[Dict[str, object]] = []
    for r in rows:
        rel = r.get("step_log_path", "")
        if not rel:
            continue
        p = ROOT / rel
        label = f"{r.get('dataset_label','run')} seed {r.get('seed','?')}"
        logs.extend(read_step_log(p, label))
    return logs


def fig_training_dynamics(rows: List[Dict[str, str]]) -> List[Path]:
    logs = collect_step_logs(rows)
    if not logs:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.axis("off")
        ax.text(0.02, 0.65, "Training dynamics unavailable", fontsize=15, weight="bold")
        ax.text(0.02, 0.45, "No referenced step_log.csv files were found in the local working tree.", fontsize=10)
        ax.text(0.02, 0.30, "Run experiments with --save-step-log and keep runs/memory_model/.../step_log.csv to enable this figure.", fontsize=10)
        return savefig("fig_adv_05_training_dynamics")
    metrics = [("exact", "Exact accuracy", (0, 1.05)), ("writer", "Writer recall / true write rate", (0, 1.05)), ("ce", "Answer CE", None)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.5, 7.0), sharex=True)
    labels = sorted({str(x["label"]) for x in logs})
    for ax, (key, title, ylim) in zip(axes, metrics):
        for label in labels:
            xs = [x["step"] for x in logs if x["label"] == label and x.get(key) is not None]
            ys = [x[key] for x in logs if x["label"] == label and x.get(key) is not None]
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Training step")
    axes[0].set_title("Training dynamics from retained step logs")
    axes[0].legend(frameon=False, fontsize=7, ncols=2, loc="upper left", bbox_to_anchor=(0, 1.28))
    return savefig("fig_adv_05_training_dynamics")


def write_manifest(paths_by_name: Dict[str, List[Path]]) -> Path:
    out = FIG_DIR / "advanced_figure_manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure_id", "png", "svg", "pdf"])
        for name, paths in paths_by_name.items():
            d = {p.suffix.lstrip("."): str(p.relative_to(ROOT)) for p in paths}
            w.writerow([name, d.get("png", ""), d.get("svg", ""), d.get("pdf", "")])
    return out


def write_audit(rows: List[Dict[str, str]], stats_path: Path, manifest_path: Path) -> Path:
    out = FIG_DIR / "advanced_figure_audit.md"
    grouped = group_rows(rows)
    lines = [
        "# Advanced figure audit",
        "",
        "Generated by `scripts/make_advanced_research_figures.py`.",
        "",
        "## Design rules used",
        "",
        "- Show individual seed points whenever seed-level data exists.",
        "- Label sample size (`n`) directly on summary plots.",
        "- Use sample SD for error bars only when `n >= 2`; do not imply confidence intervals from tiny seed counts.",
        "- Keep local-baseline writer metrics out of writer-mechanism claims.",
        "- Export PNG, SVG, and PDF for each figure.",
        "",
        "## Data included",
        "",
    ]
    for label, rs in grouped.items():
        files = sorted({r.get("source_file", "") for r in rs})
        lines.append(f"- `{label}`: n={len(rs)}, sources={', '.join(files)}")
    lines.extend([
        "",
        "## Generated files",
        "",
        f"- Stats: `{stats_path.relative_to(ROOT)}`",
        f"- Manifest: `{manifest_path.relative_to(ROOT)}`",
        "",
        "## Caveats",
        "",
        "- 2048 v2 fixed-config results may be preliminary if only seed 0 is present locally.",
        "- Cross-hardware wall-clock comparisons are approximate.",
        "- Synthetic KV recall does not prove general language-model capability.",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    rows = load_data()
    if not rows:
        raise SystemExit("No processed result CSVs found. Run experiments or create processed sweep files first.")
    stats_path = write_stats(rows)
    figures: Dict[str, List[Path]] = {}
    figures["fig_adv_01_exact_raw_seed_points"] = fig_exact_raw_points(rows)
    figures["fig_adv_02_writer_error_decomposition"] = fig_writer_error_decomposition(rows)
    figures["fig_adv_03_writer_success_vs_exact"] = fig_writer_vs_exact(rows)
    figures["fig_adv_04_efficiency_frontier"] = fig_efficiency_frontier(rows)
    figures["fig_adv_05_training_dynamics"] = fig_training_dynamics(rows)
    manifest_path = write_manifest(figures)
    audit_path = write_audit(rows, stats_path, manifest_path)
    print(f"wrote {stats_path.relative_to(ROOT)}")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    print(f"wrote {audit_path.relative_to(ROOT)}")
    print("wrote figures:")
    for name, paths in figures.items():
        print("-", ", ".join(str(p.relative_to(ROOT)) for p in paths))


if __name__ == "__main__":
    main()
