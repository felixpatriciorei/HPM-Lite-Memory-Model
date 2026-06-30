# HPM-Lite Repo Cleanup Plan

## Current repo state observed

The GitHub repo currently has:
- `docs/`
- `hpm_lite/`
- `results/`
- `scripts/`
- `tests/`
- `.gitignore`
- `README.md`
- `requirements.txt`

The README already contains:
- the updated project framing
- result snapshot
- learned-writer 2048 result
- architecture section
- current figures list
- limitations
- roadmap

The repo also has older diagnostics:
- `docs/results.md` with Stage 2 write-mode validation
- `docs/structured_memory_readout.md`
- validation scripts and structured readout scripts

## Keep

Keep these:
- `hpm_lite/`
- `scripts/run_memory_model.py`
- `scripts/run_validation.py`
- `scripts/run_structured_readers.py`
- `scripts/run_noisy_slot_extraction.py`
- `tests/`
- `results/oracle_distance_results.csv`
- `results/learned_writer_results.csv`
- `results/derived_statistics.csv`
- `docs/figures/hpm_lite_model_paths.png`

Reason:
These are useful as code, historical evidence, or reproducibility scaffolding.

## Keep but reframe

### `docs/results.md`

Do not delete it. It contains historical validation data and controls.

But it should be renamed or moved later:

`docs/archive/stage2_write_mode_validation.md`

Reason:
It is long and old. It should not be the main evidence page anymore.

### `docs/structured_memory_readout.md`

Keep, but move later to:

`docs/archive/structured_memory_readout.md`

or keep as:

`docs/diagnostics/structured_memory_readout.md`

Reason:
It is relevant, but not central to the current long-range KV proof.

## Remove or move

### `docs/research_figure_advice.md`

This file is meta-advice, not project evidence.

Option A:
Delete it from public repo.

Option B:
Move it to:
`docs/dev/research_figure_advice.md`

Recommendation:
Delete it or replace it with `docs/research_grade_figure_plan.md`.

## Replace

### Current simple figures

Do not permanently delete the old simple figures yet. Move them to:

`docs/figures/archive/`

or rename:
`pilot_01_exact_recall_vs_distance.png`

Reason:
They are valid pilot figures, but they should not be the final README figures.

### `scripts/make_scientific_figures.py`

Replace or extend into:

`scripts/make_research_figures.py`

Keep `make_scientific_figures.py` as a compatibility wrapper if needed.

## Add

Add these:

```text
docs/research_grade_figure_plan.md
docs/experiment_matrix.md
docs/logging_schema.md
docs/repo_cleanup_plan.md

results/README.md
results/raw/
results/processed/
results/traces/

scripts/aggregate_results.py
scripts/make_research_figures.py
scripts/run_sweep.py
```

## README recommendation

README should show only:
- short project claim
- one architecture diagram
- one main result figure
- one learned-writer result
- limitations
- how to run
- link to detailed docs

Move long detailed discussions to docs.

The README should not become the paper.
