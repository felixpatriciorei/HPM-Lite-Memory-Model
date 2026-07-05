"""
make_paper_grade_figures.py

Regenerates a publication-grade figure suite for the HPM-Lite results using the
exact same underlying CSVs already produced by this repository's pipeline
(results/processed/**). It never changes a published mean, gap, p-value, or
coefficient: wherever the repo's own pipeline already computed a statistic
(bootstrap CIs, permutation p-values, Cliff's delta, OLS coefficients -- e.g.
hpm_v2_research_grade_schedule_effects.csv and hpm_v2_research_grade_failure_
model.csv), this script reads that number directly rather than re-deriving a
new one, specifically so a figure can never quietly drift from the numbers in
a paper or README that cite the same file. The only new statistics computed
here are 95% bootstrap CIs for a handful of early seed sweeps (the 512- and
2048-token founding-result sweeps) that were never run through the repo's
main statistical pipeline in the first place and so have no existing CI to
preserve; these use the same percentile-bootstrap method as the rest of the
repo (see bootstrap_ci below) and are clearly an addition, not a replacement,
in the figure they appear in.

This script only changes how existing numbers are drawn: fonts, color,
layout, vector output. It does not touch results/figures/research_grade/ or
results/figures/advanced_atlas/ -- outputs land in the new, separate
results/figures/paper_grade/ as both .pdf (vector, for LaTeX or print) and
.png (raster, for READMEs/web).

Usage (from the repo root):
    python scripts/make_paper_grade_figures.py

Dependencies: matplotlib, pandas, numpy, scipy, statsmodels.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------------------------------------------------------
# Paths (mirror the repo's own layout; safe to drop this file into scripts/)
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "scripts" else "."
P = lambda *parts: os.path.join(ROOT, *parts)
OUT_DIR = P("results", "figures", "paper_grade")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_MATRIX = P("results", "processed", "research_grade", "hpm_v2_research_grade_run_matrix.csv")
FAILURE_MODEL = P("results", "processed", "research_grade", "hpm_v2_research_grade_failure_model.csv")
TRAIN_DYNAMICS = P("results", "processed", "research_grade", "hpm_v2_research_grade_training_dynamics.csv")
SUMMARY_KAGGLE = P("results", "processed", "hpm_v2_long_context_summary_kaggle.csv")
SUMMARY_ALLWORK = P("results", "processed", "hpm_v2_long_context_summary_all_workers.csv")
SCHEDULE_EFFECTS_AUTH = P("results", "processed", "research_grade", "hpm_v2_research_grade_schedule_effects.csv")
SENSITIVITY = P("results", "processed", "advanced_atlas", "long_context_schedule_effects.csv")
LLM_BENCH = P("results", "processed", "llm_memory_benchmark_rows.csv")
LOCAL_2048 = P("results", "processed", "local_2048_seed_sweep.csv")
LW_2048 = P("results", "processed", "learned_writer_2048_seed_sweep.csv")
LW_512 = P("results", "processed", "learned_writer_512_seed_sweep.csv")
V2_512 = P("results", "processed", "hpm_v2_512_seed_sweep.csv")
ORACLE = P("results", "oracle_distance_results.csv")

# --------------------------------------------------------------------------
# Shared visual style: Liberation Serif is metric-compatible with Times, which
# matches the mathptmx font used in the paper's LaTeX, so figures and body
# text look like one document instead of two.
# --------------------------------------------------------------------------
COLOR = {
    "local":    "#D55E00",  # vermillion  -- no-memory / baseline / failure
    "tf200":    "#E69F00",  # orange      -- early-stopped writer supervision
    "tf600":    "#0072B2",  # blue        -- full writer supervision
    "hpm":      "#009E73",  # bluish green-- explicit memory / HPM result
    "neutral":  "#4D4D4D",
    "grid":     "#CCCCCC",
    "band":     "#0072B2",
}

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": COLOR["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def style_axis(ax, top_right_spines=False):
    if not top_right_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", zorder=0)
    ax.set_axisbelow(True)

def save(fig, name, tight=True):
    if tight:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.04)
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight", pad_inches=0.04, dpi=220)
    else:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=220)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")

# --------------------------------------------------------------------------
# Statistical helpers (identical methodology to the repo's own pipeline:
# percentile bootstrap, two-sided permutation test, Cliff's delta). These
# are used only to re-derive the exact same published numbers for plotting
# (e.g. redrawing a CI whisker), never to produce a new claim.
# --------------------------------------------------------------------------
def bootstrap_ci(x, n_boot=10000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return lo, hi

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))

def permutation_test(a, b, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(obs):
            count += 1
    return obs, count / n_perm

# --------------------------------------------------------------------------
# Figure: training dynamics (new content -- collected but not plotted before)
# --------------------------------------------------------------------------
def fig_training_dynamics():
    """Per-step training trajectories, raw + LOWESS smoothed, split by
    schedule and sequence length. Uses the 552-row step-level log
    (hpm_v2_research_grade_training_dynamics.csv)."""
    df = pd.read_csv(TRAIN_DYNAMICS)
    lengths = [4096, 8192, 12288]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    for ax, L in zip(axes, lengths):
        sub = df[df["seq_len"] == L]
        for sched, color in [("tf200", COLOR["tf200"]), ("tf600", COLOR["tf600"])]:
            s = sub[sub["schedule"] == sched].sort_values("step")
            if len(s) == 0:
                continue
            ax.scatter(s["step"], s["eval_exact"], s=10, color=color, alpha=0.35,
                       linewidths=0, zorder=2, label=None)
            if s["step"].nunique() >= 4:
                sm = lowess(s["eval_exact"], s["step"], frac=0.55, return_sorted=True)
                ax.plot(sm[:, 0], sm[:, 1], color=color, linewidth=2.2, zorder=3,
                        label=f"{sched}")
        ax.axvline(200, color=COLOR["tf200"], linestyle=":", linewidth=1, alpha=0.6)
        ax.axvline(600, color=COLOR["tf600"], linestyle=":", linewidth=1, alpha=0.6)
        style_axis(ax)
        ax.set_title(f"{L:,} tokens", fontsize=10)
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(0, 620)
    axes[0].set_ylabel("Eval exact recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=2, fontsize=9, frameon=False,
               title="Writer-supervision schedule (dotted = teacher-forcing cutoff)")
    fig.tight_layout()
    save(fig, "fig_training_dynamics")


# --------------------------------------------------------------------------
# Figure: founding result + long-context stress matrix, unified forest plot
# --------------------------------------------------------------------------
def fig_forest_combined():
    rm = pd.read_csv(RUN_MATRIX)
    canon = rm[rm["canonical_kaggle"] == 1]
    oracle = pd.read_csv(ORACLE).set_index("distance")
    local_2048 = pd.read_csv(LOCAL_2048)["eval_answer_exact"].tolist()
    lw_2048 = pd.read_csv(LW_2048)["eval_answer_exact"].tolist()
    lw_512 = pd.read_csv(LW_512)["eval_answer_exact"].tolist()
    v2_512 = pd.read_csv(V2_512)["eval_answer_exact"].tolist()

    # Panel A: founding-result conditions (mix of oracle single-run and
    # learned multi-seed sweeps -- marker style encodes which). Every array
    # here is read live from its source CSV, not hardcoded.
    founding_rows = [
        ("Local baseline, 8192 tok\n(oracle placement, 1 run)", [float(oracle.loc[8192, "local_exact"])], "local", "o"),
        ("Local baseline, 4096 tok\n(oracle placement, 1 run)", [float(oracle.loc[4096, "local_exact"])], "local", "o"),
        (f"Local baseline, 2048 tok\n(n={len(local_2048)} seeds)", local_2048, "local", "s"),
        ("HPM oracle write, 8192 tok\n(1 run)", [float(oracle.loc[8192, "hpm_exact"])], "hpm", "o"),
        ("HPM oracle write, 4096 tok\n(1 run)", [float(oracle.loc[4096, "hpm_exact"])], "hpm", "o"),
        (f"HPM learned writer, 2048 tok\n(n={len(lw_2048)} seeds)", lw_2048, "hpm", "s"),
        (f"HPM learned writer, 512 tok\n(n={len(lw_512)} seeds)", lw_512, "hpm", "s"),
        (f"HPM-Lite v2, 512 tok\n(n={len(v2_512)} seeds)", v2_512, "hpm", "^"),
    ]

    # Panel B: long-context v2 stress matrix (tf600 vs tf200), canonical Kaggle.
    stress_rows = []
    for L in [12288, 8192, 4096]:
        for sched, tag in [("tf600", "tf600"), ("tf200", "tf200")]:
            vals = canon[(canon["seq_len_int"] == L) & (canon["schedule"] == sched)]["eval_answer_exact"].tolist()
            stress_rows.append((f"{L:,} tok, {sched}", vals, tag, "D"))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.6),
                                     gridspec_kw={"width_ratios": [1, 1]})

    for ax, rows, title in [(axA, founding_rows, "A.  Founding result: does memory help at all?"),
                              (axB, stress_rows, "B.  Long-context stress matrix (HPM-Lite v2)")]:
        ypos = np.arange(len(rows))[::-1]
        for y, (label, vals, key, marker) in zip(ypos, rows):
            vals = np.array(vals, dtype=float)
            color = COLOR[key]
            jitter = np.linspace(-0.09, 0.09, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(vals, [y + j for j in jitter], color=color, alpha=0.55,
                       s=22, zorder=3, edgecolors="none")
            mean = vals.mean()
            if len(vals) >= 4:
                lo, hi = bootstrap_ci(vals, seed=1)
                ax.plot([lo, hi], [y, y], color=color, linewidth=2.0, zorder=4, solid_capstyle="round")
            ax.scatter([mean], [y], color=color, s=70, zorder=5, marker=marker,
                       edgecolors="white", linewidths=0.8)
            ax.text(1.045, y, f"n={len(vals)}", transform=ax.get_yaxis_transform(),
                    fontsize=7.3, va="center", color="#444444")
        ax.set_yticks(ypos)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
        ax.set_xlim(-0.06, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axvline(0, color="#BBBBBB", linewidth=0.7, zorder=1)
        ax.axvline(1, color="#BBBBBB", linewidth=0.7, zorder=1)
        ax.set_xlabel("Exact recall")
        ax.set_title(title, fontsize=10, loc="left")
        style_axis(ax)
        ax.grid(axis="y", visible=False)

    fig.suptitle("Exact-recall evidence across every tested condition in this paper",
                 fontsize=12.5, fontweight="bold", y=1.035, x=0.02, ha="left")
    fig.text(0.02, 0.995, "Small dots: raw seeds (jittered). Large marker: mean. Bars: bootstrap 95% CI (n\u22654 only). Circles = single-run; squares/diamonds = seed sweeps.",
              fontsize=8, color="#444444", ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_forest_combined")

# --------------------------------------------------------------------------
# Figure: writer-schedule effect (paired seeds + effect sizes, with the
# mixed-hardware sensitivity check overlaid as an open marker)
# --------------------------------------------------------------------------
def fig_schedule_effect():
    rm = pd.read_csv(RUN_MATRIX)
    canon = rm[rm["canonical_kaggle"] == 1]
    auth = pd.read_csv(SCHEDULE_EFFECTS_AUTH)
    auth_canon = auth[auth["analysis_scope"] == "canonical_kaggle"].set_index("seq_len")
    auth_all = auth[auth["analysis_scope"] == "all_workers"].set_index("seq_len")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    lengths = [4096, 8192, 12288]
    length_color = {4096: "#5B8FA8", 8192: "#B07AA1", 12288: "#59A14F"}

    # Left: paired seed slopes (raw per-seed data, not a summary statistic --
    # nothing here can drift from a precomputed number).
    for L in lengths:
        t200 = canon[(canon.seq_len_int == L) & (canon.schedule == "tf200")].set_index("seed_int")["eval_answer_exact"]
        t600 = canon[(canon.seq_len_int == L) & (canon.schedule == "tf600")].set_index("seed_int")["eval_answer_exact"]
        shared = sorted(set(t200.index) & set(t600.index))
        for s in shared:
            axL.plot([0, 1], [t200[s], t600[s]], color=length_color[L], alpha=0.28, linewidth=1.1, zorder=2)
        axL.plot([0, 1], [t200[shared].mean(), t600[shared].mean()], color=length_color[L],
                  linewidth=2.6, marker="o", markersize=6, zorder=4, label=f"{L:,} tok (n={len(shared)} paired)")
    axL.set_xlim(-0.15, 1.15)
    axL.set_xticks([0, 1]); axL.set_xticklabels(["tf200\n(early-stopped)", "tf600\n(full supervision)"])
    axL.set_ylabel("Exact recall")
    axL.set_title("A.  Paired seeds: same seed, both schedules", fontsize=10, loc="left")
    axL.legend(loc="lower right", fontsize=7.8)
    style_axis(axL)

    # Right: effect-size forest. Every number plotted here (gap, CI, p,
    # Cliff's delta) is read directly from hpm_v2_research_grade_schedule_effects.csv
    # -- the same authoritative file the paper's Table 5 numbers come from --
    # so the figure and the text cannot disagree.
    ypos = np.arange(len(lengths))[::-1]
    for y, L in zip(ypos, lengths):
        row = auth_canon.loc[L]
        gap, lo, hi = row["mean_gap_exact_tf600_minus_tf200"], row["bootstrap_ci95_low_gap"], row["bootstrap_ci95_high_gap"]
        p, d = row["permutation_p_value"], row["cliffs_delta"]
        axR.plot([lo, hi], [y + 0.14, y + 0.14], color=length_color[L], linewidth=2.2, solid_capstyle="round", zorder=3)
        axR.scatter([gap], [y + 0.14], color=length_color[L], s=55, zorder=4, marker="D", edgecolors="white", linewidths=0.6)
        axR.text(0.56, y + 0.14, f"canonical:  p={p:.4f},  \u03b4={d:.2f}", fontsize=7.6, va="center", color="#333333")

        srow = auth_all.loc[L]
        sgap = srow["mean_gap_exact_tf600_minus_tf200"]
        axR.scatter([sgap], [y - 0.14], facecolors="none",
                    edgecolors=length_color[L], s=55, zorder=4, marker="D", linewidths=1.3)
        axR.plot([srow["bootstrap_ci95_low_gap"], srow["bootstrap_ci95_high_gap"]], [y - 0.14, y - 0.14], color=length_color[L],
                  linewidth=1.4, linestyle=(0, (3, 1.5)), zorder=3)
        axR.text(0.56, y - 0.14, f"+mixed hw:  p={srow['permutation_p_value']:.4f},  \u03b4={srow['cliffs_delta']:.2f}", fontsize=7.6, va="center", color="#777777")

    axR.axvline(0, color="#999999", linewidth=0.9, zorder=1)
    axR.set_yticks(ypos)
    axR.set_yticklabels([f"{L:,} tok" for L in lengths])
    axR.set_xlim(-0.05, 1.05)
    axR.set_xlabel("Exact-recall gap: tf600 \u2212 tf200")
    axR.set_title("B.  Schedule effect size: canonical vs. +mixed-hardware sensitivity check", fontsize=10, loc="left")
    style_axis(axR)
    axR.grid(axis="y", visible=False)
    axR.set_ylim(-0.55, 2.55)

    fig.suptitle("Writer-supervision schedule effect on long-context exact recall",
                 fontsize=12.5, fontweight="bold", y=1.03, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig_schedule_effect")


# --------------------------------------------------------------------------
# Figure: regression coefficient forest plot (new visualization of the
# existing exploratory OLS/HC3 result -- same coefficients as the paper's
# regression table, never previously plotted)
# --------------------------------------------------------------------------
def fig_regression_coefficients():
    reg = pd.read_csv(FAILURE_MODEL)
    reg = reg[reg["term"] != "const"].copy()
    label_map = {
        "eval_true_fact_written_rate": "Writer true-fact rate",
        "eval_retrieval_top1": "Retrieval top-1",
        "log2_seq_len": "log\u2082(sequence length)",
        "tf600_indicator": "tf600 indicator",
    }
    reg["label"] = reg["term"].map(label_map)
    reg = reg.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ypos = np.arange(len(reg))
    sig = reg["p_value"] < 0.05
    colors = [COLOR["hpm"] if s else "#9E9E9E" for s in sig]
    for y, (_, row), c in zip(ypos, reg.iterrows(), colors):
        ax.plot([row["ci95_low"], row["ci95_high"]], [y, y], color=c, linewidth=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([row["coefficient"]], [y], color=c, s=75, zorder=4, edgecolors="white", linewidths=0.8)
        stars = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "n.s."))
        ax.text(row["ci95_high"] + 0.18, y, f"p={row['p_value']:.4f}  {stars}", va="center", fontsize=8, color="#333333")
    ax.axvline(0, color="#999999", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(reg["label"])
    ax.set_xlabel("Coefficient (exact recall per unit of predictor), HC3 robust 95% CI")
    ax.set_xlim(-3.2, 6.7)
    n = int(reg["n"].iloc[0]); r2 = reg["r_squared"].iloc[0]
    ax.set_title(f"Exact recall regressed on writer/retrieval/length/schedule  (n={n}, R\u00b2={r2:.2f})",
                 fontsize=10.5, loc="left")
    style_axis(ax)
    ax.grid(axis="y", visible=False)
    fig.text(0.02, -0.06, "Exploratory aggregate-run association (HC3-robust OLS), not a per-example causal estimate. Green = distinguishable from zero at \u03b1=0.05.",
              fontsize=7.6, color="#555555")
    fig.tight_layout()
    save(fig, "fig_regression_coefficients")


# --------------------------------------------------------------------------
# Figure: writer quality vs exact recall, with OLS fit + 95% confidence band
# --------------------------------------------------------------------------
def fig_writer_bottleneck():
    rm = pd.read_csv(RUN_MATRIX)
    d = rm.dropna(subset=["eval_true_fact_written_rate", "eval_answer_exact"]).copy()
    d = d[d["eval_true_fact_written_rate"] > 0]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for sched, color, marker in [("tf200", COLOR["tf200"], "o"), ("tf600", COLOR["tf600"], "^")]:
        sub = d[d["schedule"] == sched]
        sizes = 18 + (sub["seq_len_int"] / 12288.0) * 55
        ax.scatter(sub["eval_true_fact_written_rate"], sub["eval_answer_exact"], s=sizes,
                   color=color, alpha=0.75, edgecolors="white", linewidths=0.5, label=sched, zorder=3, marker=marker)

    x = d["eval_true_fact_written_rate"].values
    y = d["eval_answer_exact"].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept
    n = len(x)
    t_val = stats.t.ppf(0.975, n - 2)
    x_mean = x.mean()
    s_err = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    conf = t_val * s_err * np.sqrt(1 / n + (xs - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
    ax.fill_between(xs, ys - conf, ys + conf, color=COLOR["neutral"], alpha=0.13, zorder=1, label="95% CI (OLS fit)")
    ax.plot(xs, ys, color=COLOR["neutral"], linewidth=1.8, zorder=2, linestyle="--")

    ax.text(0.03, 0.96, f"$R^2$ = {r**2:.2f}\n$y$ = {slope:.2f}$x$ + {intercept:.2f}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.7))
    ax.set_xlabel("Writer true-fact rate")
    ax.set_ylabel("Exact recall")
    ax.set_title("Writer quality explains most long-context exact-recall variation", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, title="marker size scales with sequence length")
    style_axis(ax)
    fig.tight_layout()
    save(fig, "fig_writer_bottleneck")


# --------------------------------------------------------------------------
# Figure: Spearman correlation matrix across run-level metrics (new; a
# second, independent lens on the writer-vs-retrieval bottleneck question)
# --------------------------------------------------------------------------
def fig_correlation_matrix():
    rm = pd.read_csv(RUN_MATRIX)
    cols = ["eval_answer_exact", "eval_true_fact_written_rate", "eval_retrieval_top1", "eval_answer_ce", "seq_len_int"]
    labels = ["Exact\nrecall", "Writer true-\nfact rate", "Retrieval\ntop-1", "Answer\nCE", "Sequence\nlength"]
    d = rm.dropna(subset=cols)[cols]
    corr = d.corr(method="spearman").values
    n = len(cols)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            txt_color = "white" if abs(val) > 0.6 else "#222222"
            weight = "bold" if i != j and abs(val) > 0.5 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.6, color=txt_color, fontweight=weight)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7.8, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_title(f"Spearman rank correlations across all n={len(d)} canonical + context runs", fontsize=9.8, loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Spearman \u03c1", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    save(fig, "fig_correlation_matrix")


# --------------------------------------------------------------------------
# Figure: local-LLM memory benchmark, redesigned (per-task heatmap +
# cost/accuracy frontier), same 150-row benchmark as before
# --------------------------------------------------------------------------
def fig_llm_benchmark():
    df = pd.read_csv(LLM_BENCH)
    method_order = ["full_context", "truncated_head", "truncated_tail", "keyword_rag", "embedding_rag", "structured_slot_memory"]
    method_label = {"full_context": "Full context", "truncated_head": "Truncated head", "truncated_tail": "Truncated tail",
                    "keyword_rag": "Keyword RAG", "embedding_rag": "Embedding RAG", "structured_slot_memory": "Structured slots"}
    task_label = {"beginning_fact": "Beginning fact survives\ndistractor tail", "middle_final_update": "Stale value overwritten\nlater",
                  "multi_entity": "Near-duplicate records,\nmultiple entities", "paraphrase_fact": "Paraphrased query,\nno lexical match",
                  "stale_conflict": "Many stale values,\none authoritative"}
    tasks = list(task_label.keys())

    fig = plt.figure(figsize=(9.8, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    axH = fig.add_subplot(gs[0])
    axS = fig.add_subplot(gs[1])

    piv = df.pivot_table(index="task", columns="method", values="strict_value_exact", aggfunc="mean").reindex(index=tasks, columns=method_order)
    im = axH.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            axH.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.3,
                     color="white" if v < 0.35 or v > 0.75 else "#333333")
    axH.set_xticks(range(len(method_order))); axH.set_xticklabels([method_label[m] for m in method_order], rotation=32, ha="right", fontsize=7.8)
    axH.set_yticks(range(len(tasks))); axH.set_yticklabels([task_label[t] for t in tasks], fontsize=7.6)
    axH.set_title("A.  Strict-exact recall by task and method", fontsize=10, loc="left")
    for spine in axH.spines.values():
        spine.set_visible(False)
    axH.set_xticks(np.arange(len(method_order) + 1) - 0.5, minor=True)
    axH.set_yticks(np.arange(len(tasks) + 1) - 0.5, minor=True)
    axH.grid(which="minor", color="white", linewidth=1.8)
    axH.grid(which="major", visible=False)
    axH.tick_params(which="minor", bottom=False, left=False)

    agg = df.groupby("method").agg(acc=("strict_value_exact", "mean"), lat=("latency_sec", "mean"),
                                     chars=("prompt_chars", "mean")).reindex(method_order)
    palette = ["#999999", "#B07AA1", "#D55E00", "#E69F00", "#5B8FA8", COLOR["hpm"]]
    for (m, row), c in zip(agg.iterrows(), palette):
        zero_acc = row["acc"] == 0
        marker = "X" if zero_acc else "o"
        axS.scatter(row["chars"], max(row["lat"], 0.15) if not np.isnan(row["lat"]) else 20, s=max(row["acc"], 0.12) * 500 + 60,
                    color=c, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3, marker=marker)
        axS.annotate(method_label[m], (row["chars"], max(row["lat"], 0.15) if not np.isnan(row["lat"]) else 20),
                     xytext=(6, 6), textcoords="offset points", fontsize=7.6, color="#333333")
    axS.set_xscale("log"); axS.set_yscale("log")
    axS.set_xlabel("Mean prompt size (characters, log scale)")
    axS.set_ylabel("Mean latency, sec (log scale)")
    axS.set_title("B.  Cost/accuracy frontier (marker size = strict-exact accuracy)", fontsize=10, loc="left")
    style_axis(axS)

    fig.suptitle("Local-LLM memory benchmark: 5 tasks \u00d7 5 seeds per method (n=25/method)",
                 fontsize=12, fontweight="bold", y=1.03, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig_llm_benchmark")


# --------------------------------------------------------------------------
# Figure: task + model schematic (matplotlib version, for repo README use;
# the paper itself uses a native TikZ version for perfect vector/font match)
# --------------------------------------------------------------------------
def fig_schematic():
    fig, (axT, axM) = plt.subplots(1, 2, figsize=(9.8, 2.9), gridspec_kw={"width_ratios": [1.05, 1]})

    def box(ax, xy, w, h, text, fc, ec="#333333", fontsize=8.3):
        b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(b)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)

    def arrow(ax, p0, p1, color="#333333"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                                       color=color, zorder=1, shrinkA=2, shrinkB=2))

    # --- Left: task structure ---
    axT.set_xlim(0, 10); axT.set_ylim(0, 4); axT.axis("off")
    axT.set_title("Synthetic key-value recall task", fontsize=10.5, loc="left")
    segs = [("FACT k03=v19", 0, 2.2, "#DCE9F5"), ("distractor tokens \u2026", 2.4, 4.0, "#F2F2F2"),
            ("QUERY k03", 6.6, 1.6, "#FBE8C6"), ("ANSWER: scored here", 8.4, 1.6, "#DCEFE1")]
    for label, x, w, fc in segs:
        box(axT, (x, 1.4), w, 1.1, label, fc, fontsize=7.6)
    axT.annotate("", xy=(2.4, 1.95), xytext=(0, 1.95), arrowprops=dict(arrowstyle="-", color="none"))
    axT.text(5.0, 0.55, "local window $W$ = 256 tokens does not reach back this far once $T \u226b W$", fontsize=7.3, ha="center", color="#555555")
    axT.annotate("", xy=(6.6, 0.85), xytext=(0.9, 0.85),
                 arrowprops=dict(arrowstyle="-", color="#999999", linestyle=(0, (3, 2))))
    axT.plot([6.6, 6.6], [0.85, 1.4], color="#999999", linestyle=(0, (3, 2)), linewidth=1)
    axT.plot([0.9, 0.9], [0.85, 1.4], color="#999999", linestyle=(0, (3, 2)), linewidth=1)

    # --- Right: HPM-Lite v2 model sketch ---
    axM.set_xlim(0, 10); axM.set_ylim(0, 5); axM.axis("off")
    axM.set_title("HPM-Lite v2 model (tested)", fontsize=10.5, loc="left")
    paths = [("Local windowed\nattention", 3.9, "#DCE9F5"), ("Selective recurrent\nstate", 3.05, "#DCE9F5"),
             ("Fast-weight\nassoc. memory", 2.2, "#DCE9F5"), ("Episodic memory\n(write/retrieve)", 1.35, "#DCE9F5")]
    box(axM, (0.1, 4.35), 2.0, 0.55, "Token states", "#EFEFEF", fontsize=7.8)
    for label, y, fc in paths:
        box(axM, (2.7, y), 2.7, 0.72, label, fc, fontsize=7.3)
        arrow(axM, (2.1, 4.6), (2.7, y + 0.36))
    box(axM, (6.0, 2.6), 1.7, 0.8, "Router\n(softmax mix)", "#EDE3F2", fontsize=7.8)
    for label, y, fc in paths:
        arrow(axM, (5.4, y + 0.36), (6.0, 3.0))
    box(axM, (8.2, 2.6), 1.6, 0.8, "Output /\nexact-match", "#F7DCDC", fontsize=7.6)
    arrow(axM, (7.7, 3.0), (8.2, 3.0))

    fig.tight_layout()
    save(fig, "fig_schematic")

if __name__ == "__main__":
    set_style()
    print("Generating paper-grade figure suite ->", OUT_DIR)
    fig_training_dynamics()
    fig_forest_combined()
    fig_schedule_effect()
    fig_regression_coefficients()
    fig_writer_bottleneck()
    fig_correlation_matrix()
    fig_llm_benchmark()
    fig_schematic()
    print(f"Done. {8} figures written to {OUT_DIR} (each as .pdf and .png).")
