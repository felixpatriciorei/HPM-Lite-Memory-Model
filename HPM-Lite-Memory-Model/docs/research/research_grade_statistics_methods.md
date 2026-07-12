# Research-grade statistics methods

This note defines the statistical and visualization methods used by `scripts/make_research_grade_figures.py`.

## Why the reset exists

The previous figure system mixed headline plots, exploratory plots, and old development plots. The reset makes one canonical figure stack that answers the core research questions:

1. Does explicit write/retrieve memory solve exact long-range KV recall?
2. Does HPM-Lite v2 remain reliable as sequence length increases?
3. Does writer-supervision schedule explain long-context failure?
4. Is failure caused by retrieval collapse or writer/value quality?
5. What is the cost/performance frontier?

## Bootstrap confidence intervals

The default interval is a percentile bootstrap 95% confidence interval over seed-level run metrics. It is used for mean exact recall, answer CE, writer rate, and schedule-effect gaps.

Interpretation rules:

* `n >= 4`: usable but still seed-limited.
* `n < 4`: exploratory only.
* Bootstrap intervals on 12288-token rows are shown for transparency but not treated as strong proof.

## Permutation tests

For `tf600` vs `tf200`, the script reports a two-sided permutation p-value for the difference in mean exact recall. This is used as a randomization-based check, not as the only evidence.

## Effect sizes

The script reports:

* mean gap: `mean(tf600 exact) - mean(tf200 exact)`;
* bootstrap CI for the mean gap;
* Cliff's delta as a non-parametric dominance effect size;
* paired seed deltas when the same seed exists in both schedules.

## LOWESS training dynamics

LOWESS is used only for diagnostic smoothing of step-level logs. Raw step points remain visible. LOWESS curves are not treated as proof by themselves.

## ECDF and distribution views

ECDFs are preferred over histograms for small seed-level result sets because every observation is directly represented. Distribution plots are used to expose seed variance rather than hide it.

## Regression and failure model

The failure-model table is exploratory. It uses aggregate-run rows, not per-example predictions, so it should not be read as causal proof. The point is to check whether writer rate, retrieval top-1, length, and schedule align with exact recall.

## Claim-facing vs exploratory

Claim-facing:

* canonical Kaggle rows with sufficient seeds;
* raw seed points;
* bootstrap CIs;
* effect sizes;
* permutation tests.

Exploratory:

* all-worker mixed hardware sensitivity;
* 12288 low-n rows;
* LOWESS curves;
* regression coefficients;
* pairgrid correlations.

## References used for method choices

* SciPy `bootstrap` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
* SciPy `permutation_test` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
* Statsmodels LOWESS documentation: https://www.statsmodels.org/devel/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
* Seaborn ECDF documentation: https://seaborn.pydata.org/generated/seaborn.ecdfplot.html
* Seaborn PairGrid documentation: https://seaborn.pydata.org/generated/seaborn.PairGrid.html
* Plotly parallel-coordinates documentation: https://plotly.com/python/parallel-coordinates-plot/
* DABEST / estimation-statistics discussion: https://acclab.github.io/DABEST-python/blog/posts/robust-beautiful/robust-beautiful.html
* Henderson et al., Deep Reinforcement Learning That Matters: https://arxiv.org/abs/1709.06560
