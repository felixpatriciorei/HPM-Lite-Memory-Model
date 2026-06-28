# README design notes

This README rewrite follows a stricter rule: do not sell the project harder than the evidence allows.

## What changed

- Replaced older single-run framing with processed 2048 seed sweeps.
- Added matched-seed HPM-vs-local table.
- Included raw seed values and standard deviations.
- Added parameter-matching caveat.
- Avoided interpreting local baseline writer columns.
- Removed claims that imply general language ability or production readiness.
- Kept the tone human and direct rather than corporate.

## Source-of-truth data

The 2048 result section is based on:

- `results/processed/learned_writer_2048_seed_sweep.csv`
- `results/processed/local_2048_seed_sweep.csv`

## Suggested GitHub About description

A compact PyTorch research prototype testing episodic memory for long-range key-value recall against a fixed-window local Transformer baseline.

## Suggested topics

```text
pytorch
memory-model
long-context
episodic-memory
synthetic-benchmark
transformer-baseline
machine-learning
research-prototype
```

## Suggested pinned / short social description

HPM-Lite Memory Model is a small, reproducible experiment showing that explicit episodic memory can solve synthetic long-range key-value recall at 2048 tokens where a fixed-window local Transformer baseline fails.

## Warning

Do not add badges for CI, license, paper, DOI, or package release unless those things actually exist.
