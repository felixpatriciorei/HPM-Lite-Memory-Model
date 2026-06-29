#!/usr/bin/env python3
"""
Research-grade figure reset for HPM-Lite.

This script replaces the old paper/advanced figure workflow with a single,
auditable statistical figure pipeline:

- seed-level raw points, not only bars
- bootstrap confidence intervals
- permutation tests for writer schedule effects
- Cliff's delta effect size
- LOWESS training dynamics when statsmodels is available
- ECDF / distribution views
- failure-mode plots separating retrieval from writer quality
- cost/performance Pareto plots
- optional Plotly parallel-coordinates HTML

It is intentionally robust across Matplotlib versions and degrades gracefully if
optional packages are missing. The static figures are generated with Matplotlib
only; seaborn/statsmodels/plotly add richer outputs when installed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
RAW = ROOT / "results" / "raw"
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "figures" / "research_grade"
OUT_PROC = PROCESSED / "research_grade"

RNG = np.random.default_rng(12345)

# Quiet, paper-ish defaults. Colors are deliberately consistent across the suite.
COLORS = {
    "hpm": "#2E6FBB",
    "local": "#A23E48",
    "tf600": "#2A9D8F",
    "tf200": "#E76F51",
    "v1": "#264653",
    "pc": "#6C5CE7",
    "kaggle": "#2A9D8F",
    "neutral": "#5B6770",
    "grid": "#D9DEE5",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.alpha": 0.5,
    "grid.linewidth": 0.8,
})


def _safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def savefig(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values: Iterable[float], statistic: Callable[[np.ndarray], float] = np.mean,
                 confidence: float = 0.95, n_resamples: int = 20000) -> tuple[float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    # Tiny-n bootstrap intervals are exploratory. Percentile bootstrap is stable and dependency-light.
    idx = RNG.integers(0, len(x), size=(n_resamples, len(x)))
    reps = np.apply_along_axis(statistic, 1, x[idx])
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha)))


def bootstrap_gap_ci(a: Iterable[float], b: Iterable[float], n_resamples: int = 20000) -> tuple[float, float]:
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    if len(a) == 1 and len(b) == 1:
        return (float(a[0] - b[0]), float(a[0] - b[0]))
    ia = RNG.integers(0, len(a), size=(n_resamples, len(a)))
    ib = RNG.integers(0, len(b), size=(n_resamples, len(b)))
    reps = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975)))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(float(x > y) for x in a for y in b)
    lt = sum(float(x < y) for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def permutation_p_value(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if scipy_stats is not None and hasattr(scipy_stats, "permutation_test"):
        def stat(x, y, axis=None):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        try:
            res = scipy_stats.permutation_test((a, b), stat, n_resamples=9999,
                                               alternative="two-sided", random_state=12345)
            return float(res.pvalue)
        except TypeError:
            pass
    # Fallback manual permutation.
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    reps = 9999
    for _ in range(reps):
        RNG.shuffle(pooled)
        gap = abs(pooled[:len(a)].mean() - pooled[len(a):].mean())
        count += gap >= obs
    return float((count + 1) / (reps + 1))


def lowess_xy(x: np.ndarray, y: np.ndarray, frac: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        idx = np.argsort(x)
        return x[idx], y[idx]
    if sm is not None:
        try:
            res = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
            return res[:, 0], res[:, 1]
        except Exception:
            pass
    idx = np.argsort(x)
    return x[idx], pd.Series(y[idx]).rolling(3, min_periods=1, center=True).mean().to_numpy()


def load_long_context() -> pd.DataFrame:
    p = PROCESSED / "hpm_v2_long_context_matrix.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Apply the long-context import overlay first.")
    df = pd.read_csv(p)
    for c in ["seq_len", "tf_steps", "seed", "batch_size", "eval_answer_exact", "eval_answer_ce",
              "eval_retrieval_top1", "eval_true_fact_written_rate", "eval_false_write_rate",
              "eval_missed_fact_rate", "train_wall_time_sec", "examples_per_sec_recent", "peak_vram_mb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["schedule"] = df.get("schedule", df["tf_steps"].map(lambda x: f"tf{int(x)}" if pd.notna(x) else "unknown"))
    df["scope"] = np.where(df.get("canonical_kaggle", 0).astype(str).isin(["1", "True", "true"]), "canonical_kaggle", "all_workers")
    df["worker"] = df.get("compute_source", "unknown")
    df["run_label"] = df.apply(lambda r: f"{int(r.seq_len)} {r.schedule} s{int(r.seed)} {r.worker}", axis=1)
    return df


def load_baseline() -> pd.DataFrame:
    frames = []
    mapping = [
        ("learned_writer_2048_seed_sweep.csv", "HPM-Lite v1 learned writer"),
        ("local_2048_seed_sweep.csv", "Local Transformer baseline"),
        ("hpm_v2_512_seed_sweep.csv", "HPM-Lite v2 512 sanity"),
        ("hpm_v2_2048_tf600_lw03_seed_sweep.csv", "HPM-Lite v2 2048 tf600"),
    ]
    for fn, label in mapping:
        p = PROCESSED / fn
        if p.exists():
            tmp = pd.read_csv(p)
            tmp["condition"] = label
            for c in ["seq_len", "seed", "eval_answer_exact", "eval_answer_ce", "eval_retrieval_top1",
                      "eval_true_fact_written_rate", "eval_false_write_rate", "eval_missed_fact_rate",
                      "train_wall_time_sec", "peak_vram_mb"]:
                if c in tmp.columns:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_step_logs(matrix: pd.DataFrame) -> pd.DataFrame:
    records = []
    # Prefer imported logs so the script doesn't depend on huge runs/ directories.
    candidates = list((LOGS / "long_context_import").glob("hpm_v2_*_lw03_bs*.log"))
    candidates += list(LOGS.glob("hpm_v2_*_lw03_bs*_pc.log"))
    pattern = re.compile(r"hpm_v2_(\d+)_seed(\d+)_tf(\d+)_lw03_bs(\d+)(?:_pc)?\.log")
    for path in candidates:
        m = pattern.search(path.name)
        if not m:
            continue
        seq_len, seed, tf, bs = map(int, m.groups())
        worker = "pc" if path.name.endswith("_pc.log") else "kaggle"
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append({
                    "seq_len": seq_len,
                    "seed": seed,
                    "tf_steps": tf,
                    "schedule": f"tf{tf}",
                    "batch_size": bs,
                    "worker": worker,
                    "step": obj.get("step"),
                    "eval_exact": obj.get("eval_exact"),
                    "eval_ce": obj.get("eval_ce"),
                    "eval_ret_top1": obj.get("eval_ret_top1"),
                    "loss": obj.get("loss"),
                    "writer_recall": obj.get("writer_recall"),
                    "log_file": str(path.relative_to(ROOT)),
                })
    df = pd.DataFrame(records)
    if not df.empty:
        for c in ["seq_len", "seed", "tf_steps", "batch_size", "step", "eval_exact", "eval_ce", "eval_ret_top1", "loss", "writer_recall"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def make_inference_tables(long_df: pd.DataFrame, base_df: pd.DataFrame, step_df: pd.DataFrame) -> None:
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    run = long_df.copy()
    run.to_csv(OUT_PROC / "hpm_v2_research_grade_run_matrix.csv", index=False)

    rows = []
    for scope_name, sdf in [("canonical_kaggle", run[run["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]),
                            ("all_workers", run)]:
        for (seq, sched), g in sdf.groupby(["seq_len", "schedule"]):
            exact = g["eval_answer_exact"].dropna().to_numpy(float)
            ce = g["eval_answer_ce"].dropna().to_numpy(float)
            writer = g["eval_true_fact_written_rate"].dropna().to_numpy(float)
            retrieval = g["eval_retrieval_top1"].dropna().to_numpy(float)
            vram = g["peak_vram_mb"].dropna().to_numpy(float)
            wall = g["train_wall_time_sec"].dropna().to_numpy(float)
            exps = g["examples_per_sec_recent"].dropna().to_numpy(float)
            eci = bootstrap_ci(exact)
            cci = bootstrap_ci(ce)
            wci = bootstrap_ci(writer)
            rows.append({
                "analysis_scope": scope_name,
                "seq_len": int(seq),
                "schedule": sched,
                "tf_steps": int(str(sched).replace("tf", "")) if str(sched).startswith("tf") else "",
                "n": len(exact),
                "seeds": ",".join(map(str, sorted(set(g["seed"].dropna().astype(int))))),
                "mean_exact": np.nanmean(exact) if len(exact) else np.nan,
                "median_exact": np.nanmedian(exact) if len(exact) else np.nan,
                "sd_exact": np.nanstd(exact, ddof=1) if len(exact) > 1 else 0.0,
                "ci95_low_exact": eci[0],
                "ci95_high_exact": eci[1],
                "mean_ce": np.nanmean(ce) if len(ce) else np.nan,
                "median_ce": np.nanmedian(ce) if len(ce) else np.nan,
                "ci95_low_ce": cci[0],
                "ci95_high_ce": cci[1],
                "mean_writer_true_fact": np.nanmean(writer) if len(writer) else np.nan,
                "median_writer_true_fact": np.nanmedian(writer) if len(writer) else np.nan,
                "ci95_low_writer": wci[0],
                "ci95_high_writer": wci[1],
                "mean_retrieval_top1": np.nanmean(retrieval) if len(retrieval) else np.nan,
                "mean_vram_mb": np.nanmean(vram) if len(vram) else np.nan,
                "mean_wall_time_sec": np.nanmean(wall) if len(wall) else np.nan,
                "mean_examples_per_sec": np.nanmean(exps) if len(exps) else np.nan,
                "compute_sources": ",".join(sorted(set(g.get("compute_source", pd.Series(["unknown"])).astype(str)))),
                "note": "claim-safe" if scope_name == "canonical_kaggle" and len(exact) >= 4 else "exploratory/low-n or mixed workers",
            })
    summary = pd.DataFrame(rows).sort_values(["analysis_scope", "seq_len", "schedule"])
    summary.to_csv(OUT_PROC / "hpm_v2_research_grade_inference_summary.csv", index=False)

    eff_rows = []
    for scope_name, sdf in [("canonical_kaggle", run[run["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]),
                            ("all_workers", run)]:
        for seq, g in sdf.groupby("seq_len"):
            a = g[g["schedule"] == "tf600"]
            b = g[g["schedule"] == "tf200"]
            if len(a) == 0 or len(b) == 0:
                continue
            av = a["eval_answer_exact"].dropna().to_numpy(float)
            bv = b["eval_answer_exact"].dropna().to_numpy(float)
            gap = float(np.nanmean(av) - np.nanmean(bv))
            gci = bootstrap_gap_ci(av, bv)
            common = sorted(set(a["seed"].dropna().astype(int)).intersection(set(b["seed"].dropna().astype(int))))
            paired = []
            for seed in common:
                aa = a[a["seed"].astype(int) == seed]["eval_answer_exact"].dropna()
                bb = b[b["seed"].astype(int) == seed]["eval_answer_exact"].dropna()
                if len(aa) and len(bb):
                    paired.append(float(aa.iloc[0] - bb.iloc[0]))
            eff_rows.append({
                "analysis_scope": scope_name,
                "seq_len": int(seq),
                "n_tf600": len(av),
                "n_tf200": len(bv),
                "mean_exact_tf600": np.nanmean(av),
                "mean_exact_tf200": np.nanmean(bv),
                "mean_gap_exact_tf600_minus_tf200": gap,
                "bootstrap_ci95_low_gap": gci[0],
                "bootstrap_ci95_high_gap": gci[1],
                "permutation_p_value": permutation_p_value(av, bv),
                "cliffs_delta": cliffs_delta(av, bv),
                "paired_seed_count": len(paired),
                "paired_mean_gap": np.mean(paired) if paired else np.nan,
                "paired_min_gap": np.min(paired) if paired else np.nan,
                "paired_max_gap": np.max(paired) if paired else np.nan,
                "claim_status": "claim-safe" if scope_name == "canonical_kaggle" and len(av) >= 4 and len(bv) >= 4 else "exploratory/low-n or mixed workers",
            })
    effects = pd.DataFrame(eff_rows).sort_values(["analysis_scope", "seq_len"])
    effects.to_csv(OUT_PROC / "hpm_v2_research_grade_schedule_effects.csv", index=False)

    # Exploratory regression summary.
    reg_rows = []
    try:
        reg = run[["eval_answer_exact", "eval_true_fact_written_rate", "eval_retrieval_top1", "seq_len", "tf_steps"]].dropna().copy()
        reg["log2_seq_len"] = np.log2(reg["seq_len"].astype(float))
        reg["tf600_indicator"] = (reg["tf_steps"].astype(int) == 600).astype(float)
        y = reg["eval_answer_exact"].astype(float)
        X = reg[["eval_true_fact_written_rate", "eval_retrieval_top1", "log2_seq_len", "tf600_indicator"]].astype(float)
        if sm is not None and len(reg) >= X.shape[1] + 3:
            Xc = sm.add_constant(X)
            model = sm.OLS(y, Xc).fit(cov_type="HC3")
            for name in model.params.index:
                ci = model.conf_int().loc[name].tolist()
                reg_rows.append({
                    "model": "OLS exact ~ writer + retrieval + length + tf600, HC3 robust SE",
                    "term": name,
                    "coefficient": model.params[name],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "p_value": model.pvalues[name],
                    "n": int(model.nobs),
                    "r_squared": model.rsquared,
                    "note": "exploratory aggregate-run regression; not causal proof",
                })
        else:
            corr = reg[["eval_answer_exact", "eval_true_fact_written_rate", "eval_retrieval_top1", "log2_seq_len", "tf600_indicator"]].corr(numeric_only=True)
            corr.to_csv(OUT_PROC / "hpm_v2_research_grade_correlation_matrix.csv")
    except Exception as e:
        reg_rows.append({"model": "regression_failed", "term": "error", "coefficient": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "n": 0, "r_squared": np.nan, "note": str(e)})
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv(OUT_PROC / "hpm_v2_research_grade_failure_model.csv", index=False)

    if not step_df.empty:
        step_df.to_csv(OUT_PROC / "hpm_v2_research_grade_training_dynamics.csv", index=False)


def fig_01_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_title("HPM-Lite research-grade memory testbed: explicit write/retrieve under long-range recall", loc="left", fontsize=15, weight="bold")

    def box(x, y, w, h, text, color="#EFF4FA", edge="#31475E", fontsize=10):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                               linewidth=1.4, facecolor=color, edgecolor=edge)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
        return patch

    def arrow(x1, y1, x2, y2, text=None, rad=0.0):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                              linewidth=1.2, color="#333333", connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arr)
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2+0.15, text, ha="center", va="center", fontsize=8, color="#333333")

    box(0.4, 5.7, 2.1, 1.0, "FACT k03 v19\n... noise ...", "#F2F7FF")
    box(0.4, 3.9, 2.1, 1.0, "QUERY k03", "#FFF8E8")
    box(0.4, 2.1, 2.1, 1.0, "ANSWER v19\nscored here", "#EEF8F1")

    box(3.2, 5.8, 2.2, 0.9, "Local path\nnearby mixing", "#EEF4FF")
    box(3.2, 4.5, 2.2, 0.9, "Selective recurrent\nstream state", "#EEF4FF")
    box(3.2, 3.2, 2.2, 0.9, "Fast-weight\nassociative memory", "#EEF4FF")
    box(3.2, 1.9, 2.2, 0.9, "Episodic memory\nkey-value retrieve", "#EEF4FF")

    box(6.1, 4.0, 2.2, 1.0, "Router\npath mixture", "#F4EFFB")
    box(8.9, 4.0, 2.3, 1.0, "Output head\nP(answer token)", "#F3F7EA")
    box(11.8, 4.0, 1.7, 1.0, "Exact / CE\nmetrics", "#FFF0F0")

    for y in [6.25, 4.95, 3.65, 2.35]:
        arrow(5.4, y, 6.1, 4.5)
    arrow(2.5, 6.2, 3.2, 6.25, "tokens")
    arrow(2.5, 4.4, 3.2, 4.95)
    arrow(2.5, 2.6, 3.2, 2.35)
    arrow(8.3, 4.5, 8.9, 4.5)
    arrow(11.2, 4.5, 11.8, 4.5)

    ax.text(3.2, 0.85, "Diagnostic target: if retrieval top-1 is saturated but exact recall drops, inspect writer quality, CE, and schedule.",
            fontsize=10, color="#303B45")
    ax.text(9.0, 2.15, "Statistical reset: raw seed points + bootstrap CIs + effect sizes + permutation tests + training dynamics.",
            fontsize=10, color="#303B45")
    savefig(fig, "fig_rg_01_model_task_schematic")


def fig_02_main_claim_forest(long_df: pd.DataFrame, base_df: pd.DataFrame) -> None:
    rows = []
    if not base_df.empty:
        for cond, g in base_df.groupby("condition"):
            vals = g["eval_answer_exact"].dropna().to_numpy(float)
            if len(vals):
                rows.append({"label": cond, "group": "baseline", "values": vals, "n": len(vals)})
    kag = long_df[long_df["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]
    for (seq, sched), g in kag.groupby(["seq_len", "schedule"]):
        vals = g["eval_answer_exact"].dropna().to_numpy(float)
        rows.append({"label": f"HPM-Lite v2 {int(seq)} {sched}", "group": sched, "values": vals, "n": len(vals)})
    # order manually
    order = ["Local Transformer baseline", "HPM-Lite v1 learned writer", "HPM-Lite v2 512 sanity",
             "HPM-Lite v2 4096 tf200", "HPM-Lite v2 4096 tf600", "HPM-Lite v2 8192 tf200", "HPM-Lite v2 8192 tf600", "HPM-Lite v2 12288 tf200", "HPM-Lite v2 12288 tf600"]
    rows = sorted(rows, key=lambda r: order.index(r["label"]) if r["label"] in order else 999)

    fig, ax = plt.subplots(figsize=(10.5, max(5.0, len(rows)*0.55)))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        vals = r["values"]
        mean = float(np.mean(vals))
        ci = bootstrap_ci(vals)
        color = COLORS["local"] if "Local" in r["label"] else (COLORS["tf200"] if "tf200" in r["label"] else COLORS["tf600"] if "tf600" in r["label"] else COLORS["hpm"])
        ax.errorbar(mean, i, xerr=[[mean-ci[0]], [ci[1]-mean]], fmt="o", color=color, capsize=4, markersize=7)
        jitter = (RNG.random(len(vals)) - 0.5) * 0.22
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=28, color=color, alpha=0.65, edgecolor="white", linewidth=0.5)
        ax.text(1.025, i, f"n={r['n']}  mean={mean:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Exact recall")
    ax.set_xlim(-0.04, 1.18)
    ax.set_title("Main claim forest plot: exact recall with raw seeds and bootstrap intervals", loc="left", weight="bold")
    ax.axvline(1.0, color="#BBBBBB", linewidth=1, linestyle="--")
    ax.text(0.0, -0.9, "Intervals are percentile bootstrap 95% CIs over seeds; tiny-n 12288 rows are exploratory.", fontsize=9, color="#4A5560")
    ax.invert_yaxis()
    savefig(fig, "fig_rg_02_exact_claim_forest")


def fig_03_schedule_effects(long_df: pd.DataFrame) -> None:
    effects = pd.read_csv(OUT_PROC / "hpm_v2_research_grade_schedule_effects.csv")
    eff = effects[effects["analysis_scope"] == "canonical_kaggle"].copy()
    fig = plt.figure(figsize=(12, 6.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    kag = long_df[long_df["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]
    for seq, g in kag.groupby("seq_len"):
        g2 = g[g["schedule"].isin(["tf200", "tf600"])]
        for seed, sg in g2.groupby("seed"):
            if set(sg["schedule"]) >= {"tf200", "tf600"}:
                y200 = float(sg[sg["schedule"] == "tf200"]["eval_answer_exact"].iloc[0])
                y600 = float(sg[sg["schedule"] == "tf600"]["eval_answer_exact"].iloc[0])
                offset = {4096:-0.18, 8192:0.0, 12288:0.18}.get(int(seq),0)
                ax1.plot([0+offset, 1+offset], [y200, y600], color="#9AA3AD", alpha=0.6, linewidth=1)
                ax1.scatter([0+offset, 1+offset], [y200, y600], s=28, color=COLORS["neutral"], zorder=3)
        means = g2.groupby("schedule")["eval_answer_exact"].mean()
        if "tf200" in means and "tf600" in means:
            offset = {4096:-0.18, 8192:0.0, 12288:0.18}.get(int(seq),0)
            ax1.plot([0+offset, 1+offset], [means["tf200"], means["tf600"]], linewidth=3, marker="o", label=f"{int(seq)} mean")
    ax1.set_xticks([0,1])
    ax1.set_xticklabels(["tf200", "tf600"])
    ax1.set_ylabel("Exact recall")
    ax1.set_title("Paired seed slopes where schedules share seeds", loc="left", weight="bold")
    ax1.set_ylim(0.45, 1.04)
    ax1.legend(frameon=False, loc="lower right")

    y = np.arange(len(eff))
    for i, r in eff.sort_values("seq_len").reset_index(drop=True).iterrows():
        gap = r["mean_gap_exact_tf600_minus_tf200"]
        lo = r["bootstrap_ci95_low_gap"]
        hi = r["bootstrap_ci95_high_gap"]
        ax2.errorbar(gap, i, xerr=[[gap-lo], [hi-gap]], fmt="o", color=COLORS["tf600"], capsize=4, markersize=7)
        ax2.text(0.36, i, f"p={r['permutation_p_value']:.3f}, Cliff Δ={r['cliffs_delta']:.2f}", va="center", fontsize=8)
    ax2.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{int(x)} tokens" for x in eff.sort_values("seq_len")["seq_len"]])
    ax2.set_xlabel("Exact gap: tf600 − tf200")
    ax2.set_xlim(-0.02, 0.55)
    ax2.set_title("Effect-size estimates with bootstrap CIs", loc="left", weight="bold")
    fig.suptitle("Writer schedule effect: full-run supervision protects long-context recall", x=0.02, ha="left", fontsize=14, weight="bold")
    savefig(fig, "fig_rg_03_writer_schedule_estimation")


def fig_04_writer_scatter(long_df: pd.DataFrame) -> None:
    df = long_df.copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for sched, color in [("tf200", COLORS["tf200"]), ("tf600", COLORS["tf600"] )]:
        g = df[df["schedule"] == sched]
        sizes = np.interp(g["seq_len"], [g["seq_len"].min(), g["seq_len"].max()], [50, 140]) if len(g) else []
        ax.scatter(g["eval_true_fact_written_rate"], g["eval_answer_exact"], s=sizes, color=color, alpha=0.78,
                   edgecolor="white", linewidth=0.7, label=sched)
    x = df["eval_true_fact_written_rate"].to_numpy(float)
    y = df["eval_answer_exact"].to_numpy(float)
    lx, ly = lowess_xy(x, y, frac=0.55)
    ax.plot(lx, ly, color="#1F2933", linewidth=2.2, label="LOWESS trend")
    ax.set_xlabel("True fact written rate")
    ax.set_ylabel("Exact recall")
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.45, 1.04)
    ax.set_title("Writer quality explains most long-context exact-recall variation", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.text(0.37, 0.995, "Point size ≈ sequence length; trend is exploratory aggregate-run LOWESS.", fontsize=8.5, color="#4A5560")
    savefig(fig, "fig_rg_04_writer_quality_vs_exact")


def fig_05_retrieval_saturated_failure(long_df: pd.DataFrame) -> None:
    df = long_df.copy()
    df["exact_failure"] = 1.0 - df["eval_answer_exact"]
    df["writer_error"] = df[["eval_false_write_rate", "eval_missed_fact_rate"]].mean(axis=1)
    sat = df[df["eval_retrieval_top1"] >= 0.99].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, ycol, ylabel in [(axes[0], "exact_failure", "Exact failure rate"), (axes[1], "eval_answer_ce", "Answer CE")]:
        for sched, color in [("tf200", COLORS["tf200"]), ("tf600", COLORS["tf600"] )]:
            g = sat[sat["schedule"] == sched]
            ax.scatter(g["writer_error"], g[ycol], s=70, color=color, alpha=0.78, edgecolor="white", linewidth=0.7, label=sched)
        lx, ly = lowess_xy(sat["writer_error"].to_numpy(float), sat[ycol].to_numpy(float), frac=0.55)
        ax.plot(lx, ly, color="#1F2933", linewidth=2)
        ax.set_xlabel("Writer error rate: mean(false write, missed fact)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " when retrieval top-1 ≥ 0.99", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    fig.suptitle("Retrieval-saturated failure analysis: remaining errors track writer quality", x=0.02, ha="left", fontsize=14, weight="bold")
    savefig(fig, "fig_rg_05_retrieval_saturated_failure")


def fig_06_training_dynamics(step_df: pd.DataFrame) -> None:
    if step_df.empty:
        return
    # Keep the figure readable: canonical Kaggle schedules, lengths with both schedules.
    df = step_df[step_df["worker"] == "kaggle"].copy()
    df = df[df["seq_len"].isin([4096, 8192, 12288])]
    metrics = [("eval_exact", "Exact recall"), ("writer_recall", "Writer recall"), ("eval_ce", "Answer CE")]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 9.5), sharex=True)
    for col, seq in enumerate([4096, 8192, 12288]):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for sched, color in [("tf200", COLORS["tf200"]), ("tf600", COLORS["tf600"] )]:
                g = df[(df["seq_len"] == seq) & (df["schedule"] == sched)]
                if g.empty:
                    continue
                # aggregate by step for a clean trend, keep faint raw points behind.
                ax.scatter(g["step"], g[metric], s=12, alpha=0.18, color=color)
                agg = g.groupby("step")[metric].mean().reset_index()
                lx, ly = lowess_xy(agg["step"].to_numpy(float), agg[metric].to_numpy(float), frac=0.45)
                ax.plot(lx, ly, color=color, linewidth=2.2, label=sched)
            if row == 0:
                ax.set_title(f"{seq} tokens", weight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(metrics)-1:
                ax.set_xlabel("Training step")
            if metric != "eval_ce":
                ax.set_ylim(0.3, 1.05)
            ax.axvline(200, color="#777777", linestyle=":", linewidth=1)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Training dynamics: early-stop writer supervision creates unstable long-context behavior", x=0.02, ha="left", fontsize=14, weight="bold")
    fig.text(0.02, 0.01, "Vertical dotted line marks tf200 teacher-forcing cutoff. Curves are LOWESS over step means; raw points are shown faintly.", fontsize=9, color="#4A5560")
    savefig(fig, "fig_rg_06_training_dynamics_lowess")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values[np.isfinite(values)])
    if len(values) == 0:
        return values, values
    y = np.arange(1, len(values)+1) / len(values)
    return values, y


def fig_07_ecdf(long_df: pd.DataFrame) -> None:
    df = long_df[long_df["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for metric, title, ax in [("eval_answer_exact", "Exact recall ECDF", axes[0]), ("eval_answer_ce", "Answer CE ECDF", axes[1])]:
        for sched, color in [("tf200", COLORS["tf200"]), ("tf600", COLORS["tf600"] )]:
            vals = df[df["schedule"] == sched][metric].dropna().to_numpy(float)
            x, y = ecdf(vals)
            ax.step(x, y, where="post", color=color, linewidth=2.2, label=sched)
            ax.scatter(x, y, color=color, s=24)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Proportion of runs ≤ x")
        ax.set_xlabel(metric.replace("eval_", "").replace("_", " "))
    axes[0].legend(frameon=False)
    fig.suptitle("Distribution view: ECDF shows every seed without binning or smoothing", x=0.02, ha="left", fontsize=14, weight="bold")
    savefig(fig, "fig_rg_07_seed_distribution_ecdf")


def fig_08_cost_pareto(long_df: pd.DataFrame) -> None:
    df = long_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, xcol, xlabel in [(axes[0], "peak_vram_mb", "Peak VRAM (MB)"), (axes[1], "train_wall_time_sec", "Train wall time (s)")]:
        for sched, color in [("tf200", COLORS["tf200"]), ("tf600", COLORS["tf600"] )]:
            g = df[df["schedule"] == sched]
            markers = {"kaggle":"o", "pc":"s"}
            for worker, gg in g.groupby("worker"):
                ax.scatter(gg[xcol], gg["eval_answer_exact"], s=np.interp(gg["seq_len"], [4096,12288], [55,150]), color=color, marker=markers.get(worker, "o"), alpha=0.78, edgecolor="white", linewidth=0.7, label=f"{sched} {worker}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Exact recall")
        ax.set_title(f"Exact recall vs {xlabel.lower()}", loc="left", weight="bold")
        ax.set_ylim(0.45, 1.04)
    # de-duplicate legend
    handles, labels = axes[1].get_legend_handles_labels()
    seen = {}
    for h,l in zip(handles, labels):
        seen.setdefault(l,h)
    axes[1].legend(seen.values(), seen.keys(), frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Systems view: accuracy/cost Pareto frontier across length and hardware", x=0.02, ha="left", fontsize=14, weight="bold")
    savefig(fig, "fig_rg_08_cost_performance_pareto")


def fig_09_heatmap(long_df: pd.DataFrame) -> None:
    df = long_df[long_df["canonical_kaggle"].astype(str).isin(["1", "True", "true"])]
    metrics = [("eval_answer_exact", "Exact"), ("eval_true_fact_written_rate", "Writer"), ("eval_retrieval_top1", "Retrieval")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        piv = df.pivot_table(index="seq_len", columns="schedule", values=metric, aggfunc="mean").sort_index()
        arr = piv[[c for c in ["tf200", "tf600"] if c in piv.columns]].to_numpy(float)
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(arr.shape[1])); ax.set_xticklabels([c for c in ["tf200", "tf600"] if c in piv.columns])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([str(int(x)) for x in piv.index])
        ax.set_title(title, weight="bold")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i,j]:.3f}", ha="center", va="center", color="white" if arr[i,j] < 0.75 else "black", fontsize=9)
    axes[0].set_ylabel("Sequence length")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Mean rate")
    fig.suptitle("Metric heatmap: retrieval remains high while writer/exact degrade under tf200", x=0.02, ha="left", fontsize=14, weight="bold")
    savefig(fig, "fig_rg_09_metric_heatmap")


def fig_10_pairgrid(long_df: pd.DataFrame) -> None:
    df = long_df.copy()
    cols = ["eval_answer_exact", "eval_answer_ce", "eval_true_fact_written_rate", "eval_retrieval_top1", "seq_len", "peak_vram_mb"]
    small = df[cols + ["schedule"]].dropna().copy()
    if small.empty:
        return
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
        g = sns.PairGrid(small, vars=cols, hue="schedule", corner=True, height=1.6)
        g.map_lower(sns.scatterplot, s=28, alpha=0.75, edgecolor="white", linewidth=0.4)
        g.map_diag(sns.histplot, element="step", stat="density", common_norm=False)
        g.add_legend()
        g.figure.suptitle("Exploratory PairGrid: aggregate-run relationships", y=1.02, x=0.04, ha="left", weight="bold")
        for ext in ["png", "svg", "pdf"]:
            g.figure.savefig(OUT / f"fig_rg_10_exploratory_pairgrid.{ext}", bbox_inches="tight")
        plt.close(g.figure)
    else:
        corr = small[cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Exploratory correlation matrix")
        savefig(fig, "fig_rg_10_exploratory_pairgrid")


def make_interactive(long_df: pd.DataFrame) -> None:
    if px is None:
        return
    html_dir = OUT / "interactive"
    html_dir.mkdir(parents=True, exist_ok=True)
    df = long_df.copy()
    dims = ["seq_len", "tf_steps", "batch_size", "eval_answer_exact", "eval_answer_ce",
            "eval_true_fact_written_rate", "eval_retrieval_top1", "peak_vram_mb", "train_wall_time_sec"]
    df2 = df[dims + ["schedule", "worker"]].dropna(subset=["eval_answer_exact"]).copy()
    if df2.empty:
        return
    fig = px.parallel_coordinates(
        df2,
        dimensions=dims,
        color="eval_answer_exact",
        labels={
            "seq_len": "Length", "tf_steps": "TF steps", "batch_size": "Batch", "eval_answer_exact": "Exact",
            "eval_answer_ce": "CE", "eval_true_fact_written_rate": "Writer", "eval_retrieval_top1": "Retrieval",
            "peak_vram_mb": "VRAM MB", "train_wall_time_sec": "Wall s",
        },
        title="HPM-Lite v2 long-context run matrix: parallel coordinates",
    )
    fig.write_html(html_dir / "hpm_v2_long_context_parallel_coordinates.html", include_plotlyjs="cdn")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(OUT / "research_grade_figure_manifest.csv", index=False)

    audit = OUT / "research_grade_figure_audit.md"
    audit.write_text("""# Research-grade figure audit

This directory is the replacement figure suite for HPM-Lite. Legacy graph directories are intentionally retired by `scripts/reset_research_grade_figures.py`.

## Design rules

1. Show raw seed-level evidence wherever possible.
2. Use bootstrap confidence intervals for means and schedule gaps.
3. Use permutation tests and effect sizes for tf600 vs tf200 comparisons.
4. Label low-n results as exploratory.
5. Separate canonical Kaggle claims from all-worker sensitivity checks.
6. Treat LOWESS and regression as exploratory diagnostics, not causal proof.
7. Export PNG, SVG, and PDF for every static figure.
8. Keep the README centered on this figure suite only.

## Generated artifacts

See `research_grade_figure_manifest.csv` for the complete file list.
""", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_PROC.mkdir(parents=True, exist_ok=True)
    long_df = load_long_context()
    base_df = load_baseline()
    step_df = parse_step_logs(long_df)
    make_inference_tables(long_df, base_df, step_df)

    fig_01_schematic()
    fig_02_main_claim_forest(long_df, base_df)
    fig_03_schedule_effects(long_df)
    fig_04_writer_scatter(long_df)
    fig_05_retrieval_saturated_failure(long_df)
    fig_06_training_dynamics(step_df)
    fig_07_ecdf(long_df)
    fig_08_cost_pareto(long_df)
    fig_09_heatmap(long_df)
    fig_10_pairgrid(long_df)
    make_interactive(long_df)
    write_manifest()

    print("wrote research-grade stats and figures")
    print(f"figures: {OUT.relative_to(ROOT)}")
    print(f"processed: {OUT_PROC.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
