#!/usr/bin/env python3
"""
Advanced research atlas for HPM-Lite.

This is the "one place" script for deeper visual/statistical analysis across
all currently committed research data:

- HPM v1/v2 seed sweeps
- HPM v2 long-context runs and training dynamics
- local LLM memory benchmark rows from LM Studio
- legacy processed result tables when available

Outputs are written to:
  results/figures/advanced_atlas/
  results/processed/advanced_atlas/

The figure suite intentionally uses advanced Matplotlib patterns from the
project's plotting notes/transcripts: GridSpec multi-panel layouts, fill-between
envelopes, annotated heat maps, dual-axis plots, 3D surfaces, log-scale
frontiers, and publication-style tick control. SciPy/statsmodels are used when
available for bootstrap intervals, permutation tests, and LOWESS smoothing.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results" / "processed"
RAW = ROOT / "results" / "raw"
FIG_OUT = ROOT / "results" / "figures" / "advanced_atlas"
PROC_OUT = PROCESSED / "advanced_atlas"
INTERACTIVE_OUT = FIG_OUT / "interactive"

RNG = np.random.default_rng(314159)

COL = {
    "embedding_rag": "#3A86FF",
    "keyword_rag": "#FB5607",
    "structured_slot_memory": "#06D6A0",
    "full_context": "#6C757D",
    "truncated_head": "#8338EC",
    "truncated_tail": "#FF006E",
    "tf600": "#2A9D8F",
    "tf200": "#E76F51",
    "hpm": "#264653",
    "local": "#A23E48",
    "grid": "#D9DEE5",
    "dark": "#222222",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 260,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COL["grid"],
    "grid.alpha": 0.55,
    "grid.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def ensure_dirs() -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PROC_OUT.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_num(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(fig: plt.Figure, name: str) -> None:
    ensure_dirs()
    for ext in ("png", "svg"):
        fig.savefig(FIG_OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def bootstrap_ci(values: Iterable[float], n: int = 10000, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def bootstrap_gap_ci(a: Iterable[float], b: Iterable[float], n: int = 10000) -> tuple[float, float]:
    a = np.asarray([float(v) for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in b if np.isfinite(v)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    ia = RNG.integers(0, len(a), size=(n, len(a)))
    ib = RNG.integers(0, len(b), size=(n, len(b)))
    gaps = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return (float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975)))


def permutation_p(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray([float(v) for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in b if np.isfinite(v)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if scipy_stats is not None and hasattr(scipy_stats, "permutation_test"):
        def stat(x, y, axis=None):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        try:
            res = scipy_stats.permutation_test((a, b), stat, n_resamples=9999, random_state=314159)
            return float(res.pvalue)
        except Exception:
            pass
    pooled = np.concatenate([a, b]).copy()
    obs = abs(a.mean() - b.mean())
    hits = 0
    reps = 9999
    for _ in range(reps):
        RNG.shuffle(pooled)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= obs
    return float((hits + 1) / (reps + 1))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray([float(v) for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in b if np.isfinite(v)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def lowess_line(x: Iterable[float], y: Iterable[float], frac: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    if len(x) < 4:
        order = np.argsort(x)
        return x[order], y[order]
    if sm is not None:
        try:
            z = sm.nonparametric.lowess(y, x, frac=frac, return_sorted=True)
            return z[:, 0], z[:, 1]
        except Exception:
            pass
    order = np.argsort(x)
    ys = pd.Series(y[order]).rolling(3, min_periods=1, center=True).mean().to_numpy()
    return x[order], ys


def clean_method_label(x: str) -> str:
    return {
        "embedding_rag": "Embedding RAG",
        "keyword_rag": "Keyword RAG",
        "structured_slot_memory": "Structured slot",
        "full_context": "Full context",
        "truncated_head": "Truncated head",
        "truncated_tail": "Truncated tail",
        "hpm_symbolic_memory": "HPM symbolic",
    }.get(str(x), str(x))


def load_llm_rows() -> pd.DataFrame:
    df = read_csv(PROCESSED / "llm_memory_benchmark_rows.csv")
    if df.empty:
        df = read_csv(PROCESSED / "llm_memory_eval_rows.csv")
        if not df.empty:
            df = df.rename(columns={"exact": "strict_value_exact"})
            if "exact_contains" not in df.columns:
                df["exact_contains"] = df["strict_value_exact"]
    if df.empty:
        return df
    df = to_num(df, ["seed", "exact_contains", "strict_value_exact", "latency_sec", "retrieved_items", "prompt_chars"])
    df["latency_ok"] = df["latency_sec"].where(df["latency_sec"] >= 0)
    return df


def summarize_llm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(["task", "method"], dropna=False)
    rows = []
    for (task, method), sub in g:
        ok = sub[sub["status"].astype(str).eq("ok")] if "status" in sub.columns else sub
        exact = sub["strict_value_exact"].astype(float)
        lo, hi = bootstrap_ci(exact)
        rows.append({
            "task": task,
            "method": method,
            "n": len(sub),
            "strict_exact_mean": exact.mean(),
            "strict_exact_ci_low": lo,
            "strict_exact_ci_high": hi,
            "ok_count": int((sub.get("status", pd.Series(["ok"] * len(sub))).astype(str) == "ok").sum()),
            "context_limit_count": int((sub.get("status", pd.Series([""] * len(sub))).astype(str) == "context_limit").sum()),
            "mean_latency_sec": ok["latency_ok"].mean(),
            "median_latency_sec": ok["latency_ok"].median(),
            "mean_prompt_chars": sub["prompt_chars"].mean(),
            "mean_retrieved_items": sub["retrieved_items"].mean(),
        })
    out = pd.DataFrame(rows).sort_values(["task", "method"])
    out.to_csv(PROC_OUT / "llm_memory_method_summary.csv", index=False)

    # Whole-benchmark method summary.
    rows = []
    for method, sub in df.groupby("method"):
        exact = sub["strict_value_exact"].astype(float)
        lo, hi = bootstrap_ci(exact)
        ok = sub[sub.get("status", pd.Series(["ok"] * len(sub))).astype(str) == "ok"]
        rows.append({
            "method": method,
            "n": len(sub),
            "strict_exact_mean": exact.mean(),
            "strict_exact_ci_low": lo,
            "strict_exact_ci_high": hi,
            "mean_latency_sec": ok["latency_ok"].mean(),
            "median_latency_sec": ok["latency_ok"].median(),
            "mean_prompt_chars": sub["prompt_chars"].mean(),
            "mean_retrieved_items": sub["retrieved_items"].mean(),
            "context_limit_count": int((sub.get("status", pd.Series([""] * len(sub))).astype(str) == "context_limit").sum()),
        })
    pd.DataFrame(rows).sort_values("method").to_csv(PROC_OUT / "llm_memory_overall_summary.csv", index=False)
    return out


def load_long_context() -> pd.DataFrame:
    df = read_csv(PROCESSED / "research_grade" / "hpm_v2_research_grade_run_matrix.csv")
    if df.empty:
        df = read_csv(PROCESSED / "hpm_v2_long_context_matrix.csv")
    if df.empty:
        return df
    num = ["seq_len", "tf_steps", "seed", "batch_size", "eval_answer_exact", "eval_answer_ce", "eval_retrieval_top1",
           "eval_true_fact_written_rate", "eval_false_write_rate", "eval_missed_fact_rate", "train_wall_time_sec",
           "examples_per_sec_recent", "eval_examples_per_sec", "peak_vram_mb", "parameters"]
    df = to_num(df, num)
    if "schedule" not in df.columns:
        df["schedule"] = df["tf_steps"].map(lambda x: f"tf{int(x)}" if pd.notna(x) else "unknown")
    if "compute_source" not in df.columns:
        df["compute_source"] = "unknown"
    return df


def summarize_long_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (seq, tf), sub in df.groupby(["seq_len", "tf_steps"]):
        exact = sub["eval_answer_exact"].astype(float)
        lo, hi = bootstrap_ci(exact)
        rows.append({
            "seq_len": int(seq),
            "tf_steps": int(tf),
            "n": len(sub),
            "exact_mean": exact.mean(),
            "exact_ci_low": lo,
            "exact_ci_high": hi,
            "answer_ce_mean": sub["eval_answer_ce"].mean(),
            "writer_true_fact_mean": sub["eval_true_fact_written_rate"].mean(),
            "retrieval_top1_mean": sub["eval_retrieval_top1"].mean(),
            "wall_time_mean_sec": sub["train_wall_time_sec"].mean(),
            "peak_vram_mean_mb": sub["peak_vram_mb"].mean(),
            "compute_sources": ",".join(sorted(set(map(str, sub["compute_source"].dropna())))),
        })
    out = pd.DataFrame(rows).sort_values(["seq_len", "tf_steps"])
    out.to_csv(PROC_OUT / "long_context_schedule_summary.csv", index=False)

    gaps = []
    for seq, s in df.groupby("seq_len"):
        a = s[s["tf_steps"] == 600]["eval_answer_exact"].astype(float)
        b = s[s["tf_steps"] == 200]["eval_answer_exact"].astype(float)
        if len(a) and len(b):
            lo, hi = bootstrap_gap_ci(a, b)
            gaps.append({
                "seq_len": int(seq),
                "tf600_n": len(a),
                "tf200_n": len(b),
                "exact_gap_tf600_minus_tf200": a.mean() - b.mean(),
                "gap_ci_low": lo,
                "gap_ci_high": hi,
                "permutation_p": permutation_p(a, b),
                "cliffs_delta": cliffs_delta(a, b),
            })
    gap_df = pd.DataFrame(gaps).sort_values("seq_len") if gaps else pd.DataFrame()
    gap_df.to_csv(PROC_OUT / "long_context_schedule_effects.csv", index=False)
    return out


def load_training() -> pd.DataFrame:
    df = read_csv(PROCESSED / "research_grade" / "hpm_v2_research_grade_training_dynamics.csv")
    if df.empty:
        df = read_csv(PROCESSED / "hpm_v2_training_dynamics_long.csv")
    if df.empty:
        return df
    return to_num(df, ["seq_len", "seed", "tf_steps", "batch_size", "step", "eval_exact", "eval_ce", "eval_ret_top1", "writer_recall", "loss"])


def load_baselines() -> pd.DataFrame:
    frames = []
    mapping = [
        ("learned_writer_2048_seed_sweep.csv", "HPM-Lite v1 learned writer"),
        ("local_2048_seed_sweep.csv", "Local Transformer baseline"),
        ("hpm_v2_512_seed_sweep.csv", "HPM-Lite v2 512"),
        ("hpm_v2_2048_tf600_lw03_seed_sweep.csv", "HPM-Lite v2 2048 tf600"),
    ]
    for filename, label in mapping:
        df = read_csv(PROCESSED / filename)
        if not df.empty:
            df["condition"] = label
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    return to_num(df, ["seq_len", "seed", "eval_answer_exact", "eval_answer_ce", "eval_retrieval_top1", "eval_true_fact_written_rate", "parameters", "peak_vram_mb"])


def plot_llm_dashboard(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    if rows.empty or summary.empty:
        return
    methods = [m for m in ["full_context", "truncated_head", "truncated_tail", "keyword_rag", "embedding_rag", "structured_slot_memory"] if m in summary["method"].unique()]
    tasks = list(summary["task"].drop_duplicates())
    fig = plt.figure(figsize=(12.5, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    mat = summary.pivot(index="task", columns="method", values="strict_exact_mean").reindex(index=tasks, columns=methods)
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_title("Strict exact recall by task and method")
    ax.set_xticks(range(len(methods)), [clean_method_label(m) for m in methods], rotation=35, ha="right")
    ax.set_yticks(range(len(tasks)), tasks)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v < 0.55 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("strict exact")

    ax = fig.add_subplot(gs[0, 1])
    overall = rows.groupby("method").agg(prompt=("prompt_chars", "mean"), exact=("strict_value_exact", "mean"), items=("retrieved_items", "mean")).reindex(methods).dropna(how="all")
    x = np.arange(len(overall))
    ax.bar(x, overall["prompt"], color=[COL.get(m, "#777777") for m in overall.index], alpha=0.90)
    ax.set_yscale("log")
    ax.set_title("Prompt budget used by each method")
    ax.set_ylabel("mean prompt chars (log scale)")
    ax.set_xticks(x, [clean_method_label(m) for m in overall.index], rotation=35, ha="right")
    for i, (m, r) in enumerate(overall.iterrows()):
        ax.text(i, r["prompt"] * 1.12, f"{r['prompt']:.0f}\n{r['items']:.0f} item", ha="center", va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ok = rows[rows["latency_sec"] >= 0]
    violin_methods = []
    data = []
    for m in methods:
        vals = ok.loc[ok["method"] == m, "latency_sec"].dropna().values
        if len(vals):
            violin_methods.append(m)
            data.append(vals)
    if data:
        parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.8)
        for body, m in zip(parts["bodies"], violin_methods):
            body.set_facecolor(COL.get(m, "#777777"))
            body.set_alpha(0.55)
            body.set_edgecolor("black")
        for key in ("cmeans", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#222222")
                parts[key].set_linewidth(1)
        ax.set_xticks(np.arange(1, len(violin_methods) + 1), [clean_method_label(m) for m in violin_methods], rotation=35, ha="right")
    ax.set_title("Latency distribution for successful calls")
    ax.set_ylabel("latency sec")
    ax.set_yscale("log")

    ax = fig.add_subplot(gs[1, 1])
    for m in methods:
        sub = rows[rows["method"] == m]
        if sub.empty:
            continue
        ax.scatter(sub["prompt_chars"], sub["latency_ok"], s=35 + 35 * sub["retrieved_items"].fillna(0),
                   alpha=0.75, label=clean_method_label(m), color=COL.get(m, "#777777"), edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prompt chars (log)")
    ax.set_ylabel("latency sec (log)")
    ax.set_title("Efficiency frontier: smaller and lower is better")
    ax.legend(loc="best", frameon=True)

    fig.suptitle("Local LLM memory benchmark: accuracy ties, compactness separates methods", fontsize=14, y=0.98)
    savefig(fig, "fig_adv_01_llm_memory_dashboard")


def plot_llm_compaction(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    piv = rows.groupby(["task", "method"]).agg(prompt=("prompt_chars", "mean"), latency=("latency_ok", "mean"), exact=("strict_value_exact", "mean"), items=("retrieved_items", "mean")).reset_index()
    tasks = sorted(piv["task"].unique())
    fig = plt.figure(figsize=(12.5, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1.0, 1.0], wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    for task in tasks:
        sub = piv[piv["task"] == task].set_index("method")
        if {"embedding_rag", "keyword_rag", "structured_slot_memory"}.issubset(sub.index):
            xs = [0, 1, 2]
            ys = [sub.loc["embedding_rag", "prompt"], sub.loc["keyword_rag", "prompt"], sub.loc["structured_slot_memory", "prompt"]]
            ax.plot(xs, ys, marker="o", linewidth=1.6, alpha=0.75, label=task)
            ax.text(2.03, ys[-1], task.replace("_", " "), va="center", fontsize=8)
    ax.set_xticks([0, 1, 2], ["Embedding\nRAG", "Keyword\nRAG", "Structured\nslot"])
    ax.set_yscale("log")
    ax.set_ylabel("mean prompt chars (log)")
    ax.set_title("Prompt compaction by task")
    ax.legend([], [], frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    ratios = []
    for task in tasks:
        sub = piv[piv["task"] == task].set_index("method")
        if {"embedding_rag", "structured_slot_memory"}.issubset(sub.index):
            ratios.append((task, sub.loc["embedding_rag", "prompt"] / sub.loc["structured_slot_memory", "prompt"], "vs embedding"))
        if {"keyword_rag", "structured_slot_memory"}.issubset(sub.index):
            ratios.append((task, sub.loc["keyword_rag", "prompt"] / sub.loc["structured_slot_memory", "prompt"], "vs keyword"))
    rdf = pd.DataFrame(ratios, columns=["task", "ratio", "comparison"])
    for k, comp in enumerate(["vs embedding", "vs keyword"]):
        s = rdf[rdf["comparison"] == comp]
        ax.scatter([k] * len(s), s["ratio"], s=70, alpha=0.8, label=comp)
        ax.plot([k - 0.17, k + 0.17], [s["ratio"].mean()] * 2, color="black", lw=2)
    ax.set_xticks([0, 1], ["slot vs\nembedding", "slot vs\nkeyword"])
    ax.set_ylabel("RAG prompt / slot prompt")
    ax.set_title("How many times smaller?")
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ax = fig.add_subplot(gs[0, 2])
    frontier = piv[piv["method"].isin(["embedding_rag", "keyword_rag", "structured_slot_memory"])]
    for m, sub in frontier.groupby("method"):
        ax.scatter(sub["prompt"], sub["latency"], s=75, color=COL.get(m, "#777"), label=clean_method_label(m), alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prompt chars")
    ax.set_ylabel("latency sec")
    ax.set_title("RAG vs slot frontier")
    ax.legend(frameon=True)
    fig.suptitle("Structured slots preserve exactness while compressing the LLM interface", fontsize=14, y=1.02)
    savefig(fig, "fig_adv_02_llm_compaction_frontier")


def plot_long_context_heatmap(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    seqs = sorted(summary["seq_len"].dropna().unique())
    tfs = sorted(summary["tf_steps"].dropna().unique())
    exact = summary.pivot(index="seq_len", columns="tf_steps", values="exact_mean").reindex(index=seqs, columns=tfs)
    writer = summary.pivot(index="seq_len", columns="tf_steps", values="writer_true_fact_mean").reindex(index=seqs, columns=tfs)
    ce = summary.pivot(index="seq_len", columns="tf_steps", values="answer_ce_mean").reindex(index=seqs, columns=tfs)
    nmat = summary.pivot(index="seq_len", columns="tf_steps", values="n").reindex(index=seqs, columns=tfs)

    fig = plt.figure(figsize=(13, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28)
    panels = [(exact, "Exact recall", "viridis", 0, 1), (writer, "Writer true-fact rate", "magma", 0, 1), (ce, "Answer cross-entropy", "rocket_r" if False else "cividis", None, None)]
    for idx, (mat, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(tfs)), [f"tf{int(x)}" for x in tfs])
        ax.set_yticks(range(len(seqs)), [str(int(x)) for x in seqs])
        if idx == 0:
            ax.set_ylabel("sequence length")
        for i in range(len(seqs)):
            for j in range(len(tfs)):
                v = mat.iloc[i, j]
                n = nmat.iloc[i, j] if i < nmat.shape[0] and j < nmat.shape[1] else np.nan
                if pd.notna(v):
                    txt = f"{v:.2f}\nn={int(n)}" if pd.notna(n) else f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                            color="white" if (title != "Answer cross-entropy" and v < 0.65) else "black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.05)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("HPM-Lite v2 long-context schedule map", fontsize=14, y=1.03)
    savefig(fig, "fig_adv_03_long_context_heatmap")


def plot_schedule_effects(gaps: pd.DataFrame) -> None:
    if gaps.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(gaps))
    ax.axvline(0, color="#222222", lw=1)
    for i, r in gaps.iterrows():
        ax.plot([r["gap_ci_low"], r["gap_ci_high"]], [y[i], y[i]], color=COL["tf600"], lw=3, solid_capstyle="round")
        ax.scatter(r["exact_gap_tf600_minus_tf200"], y[i], s=85, color=COL["tf600"], edgecolor="white", zorder=3)
        ax.text(r["gap_ci_high"] + 0.015, y[i], f"p={r['permutation_p']:.3f}, δ={r['cliffs_delta']:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, [f"{int(x)} tokens" for x in gaps["seq_len"]])
    ax.set_xlabel("Exact recall gap: tf600 − tf200")
    ax.set_title("Writer-supervision schedule effect with bootstrap intervals")
    ax.set_xlim(min(-0.05, gaps["gap_ci_low"].min() - 0.08), max(0.35, gaps["gap_ci_high"].max() + 0.22))
    ax.grid(axis="x", alpha=0.5)
    savefig(fig, "fig_adv_04_schedule_effects")


def plot_training_dynamics(train: pd.DataFrame) -> None:
    if train.empty:
        return
    fig = plt.figure(figsize=(12.5, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)
    metrics = [("eval_exact", "Evaluation exact"), ("writer_recall", "Writer recall"), ("eval_ce", "Answer CE"), ("loss", "Training loss")]
    for ax_i, (metric, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[ax_i // 2, ax_i % 2])
        for tf, color in [(200, COL["tf200"]), (600, COL["tf600"] )]:
            sub = train[train["tf_steps"] == tf]
            if sub.empty or metric not in sub.columns:
                continue
            g = sub.groupby("step")[metric].agg(["mean", "sem", "count"]).reset_index()
            g["sem"] = g["sem"].fillna(0)
            ax.plot(g["step"], g["mean"], color=color, lw=2.2, label=f"tf{tf}")
            ax.fill_between(g["step"].to_numpy(), (g["mean"] - g["sem"]).to_numpy(), (g["mean"] + g["sem"]).to_numpy(), color=color, alpha=0.18)
            lx, ly = lowess_line(sub["step"], sub[metric], frac=0.35)
            if len(lx):
                ax.plot(lx, ly, color=color, lw=1.3, ls="--", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        if metric in {"eval_exact", "writer_recall"}:
            ax.set_ylim(-0.03, 1.05)
        elif metric in {"eval_ce", "loss"}:
            ax.set_yscale("symlog", linthresh=0.01)
        ax.legend(frameon=True)
    fig.suptitle("Training dynamics: mean ± SEM with LOWESS trend", fontsize=14, y=0.99)
    savefig(fig, "fig_adv_05_training_dynamics_envelopes")


def plot_writer_phase(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for tf, marker, color in [(200, "o", COL["tf200"]), (600, "s", COL["tf600"] )]:
        sub = df[df["tf_steps"] == tf]
        if sub.empty:
            continue
        sizes = 40 + 0.012 * sub["seq_len"].fillna(4096)
        ax.scatter(sub["eval_true_fact_written_rate"], sub["eval_answer_exact"], s=sizes, marker=marker,
                   color=color, alpha=0.75, edgecolor="white", linewidth=0.6, label=f"tf{tf}")
    lx, ly = lowess_line(df["eval_true_fact_written_rate"], df["eval_answer_exact"], frac=0.5)
    if len(lx):
        ax.plot(lx, ly, color="#222222", lw=2.0, label="LOWESS")
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("writer true-fact rate")
    ax.set_ylabel("exact recall")
    ax.set_title("Writer/retrieval phase plane: failures track writer quality")
    ax.legend(frameon=True)
    ax.text(0.47, 0.95, "marker size ∝ context length", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
    savefig(fig, "fig_adv_06_writer_phase_plane")


def plot_cost_dual_axis(df: pd.DataFrame) -> None:
    if df.empty:
        return
    summary = df.groupby(["seq_len", "tf_steps"]).agg(exact=("eval_answer_exact", "mean"), wall=("train_wall_time_sec", "mean"), vram=("peak_vram_mb", "mean"), n=("eval_answer_exact", "size")).reset_index()
    fig, ax1 = plt.subplots(figsize=(9.5, 5.4))
    ax2 = ax1.twinx()
    for tf, color in [(200, COL["tf200"]), (600, COL["tf600"] )]:
        s = summary[summary["tf_steps"] == tf].sort_values("seq_len")
        ax1.plot(s["seq_len"], s["exact"], marker="o", lw=2.2, color=color, label=f"exact tf{tf}")
        ax2.plot(s["seq_len"], s["wall"] / 60.0, marker="^", lw=1.8, ls="--", color=color, alpha=0.65, label=f"minutes tf{tf}")
        for _, r in s.iterrows():
            ax1.text(r["seq_len"], r["exact"] + 0.025, f"n={int(r['n'])}", ha="center", fontsize=8, color=color)
    ax1.set_xlabel("sequence length")
    ax1.set_ylabel("mean exact recall")
    ax2.set_ylabel("mean train wall time (minutes)")
    ax1.set_ylim(0.45, 1.05)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Accuracy/cost tradeoff across context length")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)
    savefig(fig, "fig_adv_07_cost_accuracy_dual_axis")


def plot_3d_surface(df: pd.DataFrame) -> None:
    if df.empty:
        return
    s = df.groupby(["seq_len", "tf_steps"]).agg(exact=("eval_answer_exact", "mean"), writer=("eval_true_fact_written_rate", "mean"), ce=("eval_answer_ce", "mean")).reset_index()
    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    x = s["seq_len"].to_numpy(dtype=float)
    y = s["tf_steps"].to_numpy(dtype=float)
    z = s["exact"].to_numpy(dtype=float)
    try:
        surf = ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="white", linewidth=0.6, alpha=0.88)
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.12, label="exact recall")
    except Exception:
        ax.scatter(x, y, z, c=z, cmap="viridis", s=90)
    ax.scatter(x, y, z, c=z, cmap="viridis", s=55, edgecolor="black", linewidth=0.4)
    for _, r in s.iterrows():
        ax.text(r["seq_len"], r["tf_steps"], r["exact"] + 0.02, f"{r['exact']:.2f}", fontsize=8)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("writer TF steps")
    ax.set_zlabel("exact recall")
    ax.set_title("3D context-length / schedule performance surface")
    ax.view_init(elev=24, azim=-52)
    savefig(fig, "fig_adv_08_3d_context_schedule_surface")


def plot_all_research_dashboard(llm_summary: pd.DataFrame, long_summary: pd.DataFrame, baselines: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    if not baselines.empty:
        b = baselines.groupby("condition")["eval_answer_exact"].agg(["mean", "count"]).reset_index().sort_values("mean")
        y = np.arange(len(b))
        colors = [COL["local"] if "Local" in c else COL["hpm"] for c in b["condition"]]
        ax.barh(y, b["mean"], color=colors, alpha=0.9)
        ax.set_yticks(y, b["condition"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("exact recall")
        ax.set_title("Core HPM-Lite seed sweeps")
        for i, r in b.iterrows():
            ax.text(r["mean"] + 0.02, list(b.index).index(i), f"{r['mean']:.2f} n={int(r['count'])}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No baseline seed sweeps found", ha="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[0, 1])
    if not long_summary.empty:
        for tf, color in [(200, COL["tf200"]), (600, COL["tf600"] )]:
            s = long_summary[long_summary["tf_steps"] == tf].sort_values("seq_len")
            if not s.empty:
                ax.errorbar(s["seq_len"], s["exact_mean"], yerr=[s["exact_mean"] - s["exact_ci_low"], s["exact_ci_high"] - s["exact_mean"]],
                            marker="o", lw=2, capsize=3, color=color, label=f"tf{tf}")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("exact recall")
        ax.set_ylim(0.45, 1.05)
        ax.set_title("Long-context HPM-v2 schedule result")
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, "No long-context summary found", ha="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 0])
    if not llm_summary.empty:
        overall = llm_summary.groupby("method").agg(exact=("strict_exact_mean", "mean"), prompt=("mean_prompt_chars", "mean"), latency=("mean_latency_sec", "mean")).reset_index()
        keep = overall[overall["method"].isin(["embedding_rag", "keyword_rag", "structured_slot_memory", "full_context", "truncated_head", "truncated_tail"])]
        ax.scatter(keep["prompt"], keep["exact"], s=110, c=[COL.get(m, "#777") for m in keep["method"]], edgecolor="white", linewidth=0.8)
        for _, r in keep.iterrows():
            ax.text(r["prompt"] * 1.05, r["exact"], clean_method_label(r["method"]), va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("mean prompt chars")
        ax.set_ylabel("strict exact")
        ax.set_ylim(-0.05, 1.08)
        ax.set_title("Local LLM memory benchmark")
    else:
        ax.text(0.5, 0.5, "No LLM benchmark rows found", ha="center")
        ax.set_axis_off()

    ax = fig.add_subplot(gs[1, 1])
    if not long_summary.empty:
        ax.scatter(long_summary["writer_true_fact_mean"], long_summary["exact_mean"],
                   s=55 + 0.006 * long_summary["seq_len"], c=long_summary["tf_steps"], cmap="coolwarm", edgecolor="white")
        ax.set_xlabel("writer true-fact mean")
        ax.set_ylabel("exact mean")
        ax.set_xlim(0.65, 1.02)
        ax.set_ylim(0.55, 1.05)
        ax.set_title("Writer quality remains the bottleneck")
        for _, r in long_summary.iterrows():
            ax.text(r["writer_true_fact_mean"] + 0.005, r["exact_mean"], f"{int(r['seq_len'])}/tf{int(r['tf_steps'])}", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No writer summary found", ha="center")
        ax.set_axis_off()

    fig.suptitle("HPM-Lite research atlas: memory exactness, compactness, and writer control", fontsize=15, y=0.99)
    savefig(fig, "fig_adv_09_all_research_dashboard")


def make_interactive(llm_rows: pd.DataFrame, long_df: pd.DataFrame) -> None:
    if px is None:
        return
    if not llm_rows.empty:
        keep = llm_rows.copy()
        keep = keep[keep["method"].isin(["embedding_rag", "keyword_rag", "structured_slot_memory", "full_context", "truncated_head", "truncated_tail"])]
        if not keep.empty:
            fig = px.scatter(
                keep,
                x="prompt_chars",
                y="latency_ok",
                color="method",
                symbol="task",
                size="retrieved_items",
                hover_data=["task", "seed", "gold", "pred", "status", "strict_value_exact"],
                log_x=True,
                log_y=True,
                title="Local LLM memory benchmark: prompt/latency frontier",
            )
            fig.write_html(INTERACTIVE_OUT / "llm_memory_prompt_latency_frontier.html", include_plotlyjs="cdn")
    if not long_df.empty:
        cols = ["seq_len", "tf_steps", "seed", "eval_answer_exact", "eval_answer_ce", "eval_true_fact_written_rate", "eval_retrieval_top1", "peak_vram_mb", "train_wall_time_sec"]
        available = [c for c in cols if c in long_df.columns]
        if len(available) >= 4:
            fig = px.parallel_coordinates(
                long_df[available].dropna(),
                color="eval_answer_exact" if "eval_answer_exact" in available else available[-1],
                title="HPM-v2 long-context run matrix parallel coordinates",
            )
            fig.write_html(INTERACTIVE_OUT / "long_context_parallel_coordinates_advanced.html", include_plotlyjs="cdn")


def write_report(llm_summary: pd.DataFrame, long_summary: pd.DataFrame, gaps: pd.DataFrame, baselines: pd.DataFrame) -> None:
    lines = []
    lines.append("# Advanced research atlas\n")
    lines.append("This report is generated by `python scripts/make_advanced_research_atlas.py`. It reworks the current committed research CSVs into a unified visual/statistical atlas.\n")
    lines.append("## Generated figures\n")
    for f in sorted(FIG_OUT.glob("fig_adv_*.png")):
        lines.append(f"- `{f.relative_to(ROOT)}`")
    lines.append("\n## Processed outputs\n")
    for f in sorted(PROC_OUT.glob("*.csv")):
        lines.append(f"- `{f.relative_to(ROOT)}`")
    if not llm_summary.empty:
        lines.append("\n## LLM memory benchmark result\n")
        overall = pd.read_csv(PROC_OUT / "llm_memory_overall_summary.csv")
        lines.append(overall.to_markdown(index=False))
        lines.append("\nInterpretation: the RAG methods and structured slot memory can tie on exactness, while structured slots should be judged on prompt compression and retrieved-item efficiency.\n")
    if not gaps.empty:
        lines.append("\n## Long-context schedule effect\n")
        lines.append(gaps.to_markdown(index=False))
        lines.append("\nInterpretation: the tf600 writer-supervision schedule is the central long-context lever in the current data; the plots show its effect size by sequence length.\n")
    if not baselines.empty:
        lines.append("\n## Baseline seed-sweep inputs detected\n")
        b = baselines.groupby("condition")["eval_answer_exact"].agg(["count", "mean", "std"]).reset_index()
        lines.append(b.to_markdown(index=False))
    lines.append("\n## Limits\n")
    lines.append("The LLM memory benchmark still uses symbolic structured slots; it is a target behavior for a learned HPM memory controller, not proof that the learned writer already beats RAG. The advanced figures make this explicit by separating exactness from prompt budget, latency, retrieved-item count, and writer-quality diagnostics.\n")
    out = ROOT / "docs" / "figures" / "advanced_research_atlas.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    llm_rows = load_llm_rows()
    llm_summary = summarize_llm(llm_rows) if not llm_rows.empty else pd.DataFrame()
    long_df = load_long_context()
    long_summary = summarize_long_context(long_df) if not long_df.empty else pd.DataFrame()
    gaps = read_csv(PROC_OUT / "long_context_schedule_effects.csv")
    train = load_training()
    baselines = load_baselines()

    plot_llm_dashboard(llm_rows, llm_summary)
    plot_llm_compaction(llm_rows)
    plot_long_context_heatmap(long_summary)
    plot_schedule_effects(gaps)
    plot_training_dynamics(train)
    plot_writer_phase(long_df)
    plot_cost_dual_axis(long_df)
    plot_3d_surface(long_df)
    plot_all_research_dashboard(llm_summary, long_summary, baselines)
    make_interactive(llm_rows, long_df)
    write_report(llm_summary, long_summary, gaps, baselines)

    manifest = []
    for f in sorted(FIG_OUT.rglob("*")):
        if f.is_file():
            manifest.append({"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size})
    pd.DataFrame(manifest).to_csv(PROC_OUT / "advanced_atlas_manifest.csv", index=False)

    print("wrote advanced research atlas")
    print(f"figures: {FIG_OUT.relative_to(ROOT)}")
    print(f"processed: {PROC_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
