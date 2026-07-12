# Research-grade figure reset procedure

This repo now uses a single figure pipeline:

```bash
python scripts/reset_research_grade_figures.py
```

The reset script intentionally deletes these legacy graph directories:

```text
results/figures/paper
results/figures/advanced
docs/figures
```

Then it regenerates:

```text
results/figures/research_grade
results/processed/research_grade
```

## Why delete old graphs?

The old figure directories mixed earlier v1 paper figures, preliminary advanced diagnostics, and long-context exploratory plots. Keeping them all on the front page makes the repo harder to audit. The new stack preserves the important evidence in a single unified system.

## Safety

This is a Git repo. Deleting generated figures is reversible before commit:

```bash
git restore results/figures/paper results/figures/advanced docs/figures
```

After the reset, inspect:

```bash
git status
python -m pytest -q
```

Then commit with `git add -A` so the deletions and replacements are tracked together.
