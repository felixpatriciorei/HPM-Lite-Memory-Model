# Research-grade figure audit

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
