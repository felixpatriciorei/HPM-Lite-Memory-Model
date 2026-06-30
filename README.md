# HPM-Lite

[![tests](https://github.com/felixpatriciorei/HPM-Lite-Memory-Model/actions/workflows/tests.yml/badge.svg)](https://github.com/felixpatriciorei/HPM-Lite-Memory-Model/actions/workflows/tests.yml)
[![license](https://img.shields.io/github/license/felixpatriciorei/HPM-Lite-Memory-Model)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11-blue)](requirements.txt)
[![pytorch](https://img.shields.io/badge/built%20with-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![status](https://img.shields.io/badge/status-research--prototype-yellow)](#scope-and-limitations)
[![last commit](https://img.shields.io/github/last-commit/felixpatriciorei/HPM-Lite-Memory-Model)](https://github.com/felixpatriciorei/HPM-Lite-Memory-Model/commits/main)

Small PyTorch experiments for long-range **exact recall** with explicit neural memory.

HPM-Lite is a research testbed, not a language model. It asks whether a compact model can store key-value facts in memory and recover them thousands of tokens later, beyond a fixed local attention window.

<p align="center">
  <img src="results/figures/research_grade/fig_rg_02_exact_claim_forest.png" alt="Research-grade exact recall summary" width="820">
</p>

<details>
<summary><strong>Table of contents</strong></summary>

- [Why this exists](#why-this-exists)
- [Current result](#current-result)
- [Model sketch](#model-sketch)
- [What is measured](#what-is-measured)
- [Research-grade figures](#research-grade-figures)
- [More diagnostics](#more-diagnostics)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Model files](#model-files)
- [Main files](#main-files)
- [Scope and limitations](#scope-and-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [About](#about)
- [Citation status](#citation-status)
- [License](#license)

</details>

## Why this exists

Standard attention is a strong content-addressable memory while the needed tokens remain inside the context window. The hard case is different: a fact appears early, distractor tokens fill the middle, and the query arrives after the relevant fact has fallen outside the local window.

HPM-Lite isolates that problem with a synthetic key-value recall task:

```text
FACT k12 v77
FACT k03 v19
FACT k88 v41
NOISE ...
QUERY k03
ANSWER v19
```

The answer is scored only at the final answer position. The main question is not whether the model can model language; it is whether it writes the right facts, retrieves them later, and routes retrieved memory into prediction.

## Current result

On the long-context v2 stress matrix, full-run writer supervision (`tf600`) keeps exact recall high at 4096 and 8192 tokens. Early-stop writer supervision (`tf200`) drops sharply, even when retrieval top-1 stays near 1.0.

**Working interpretation:** in these runs, retrieval is mostly saturated; the remaining long-context failure mode is writer/value quality.

### Canonical Kaggle runs

These are the claim-facing Kaggle T4 runs. Intervals are percentile bootstrap 95% confidence intervals over seeds. The 12288-token rows are included as early stress evidence, not a final claim.

| Sequence length | Schedule | Seeds | Exact accuracy | 95% CI | Writer true-fact rate | Retrieval top-1 | Status |
|---:|:---|---:|---:|:---|---:|---:|:---|
| 4096 | tf600 | 8 | 0.9938 | [0.9844, 1.0000] | 0.9914 | 1.0000 | stable |
| 4096 | tf200 | 4 | 0.8000 | [0.6625, 0.9125] | 0.8539 | 1.0000 | comparison |
| 8192 | tf600 | 8 | 0.9750 | [0.9500, 0.9938] | 0.9766 | 1.0000 | stable |
| 8192 | tf200 | 6 | 0.7250 | [0.6167, 0.8250] | 0.7344 | 0.9907 | comparison |
| 12288 | tf600 | 2 | 0.9500 | [0.9000, 1.0000] | 0.9437 | 1.0000 | low-n |
| 12288 | tf200 | 2 | 0.6500 | [0.6000, 0.7000] | 0.7437 | 1.0000 | low-n |

Schedule effect, measured as `tf600 - tf200` exact accuracy:

| Sequence length | Exact gap | 95% CI | Permutation p | Cliff's delta | Status |
|---:|---:|:---|---:|---:|:---|
| 4096 | +0.1938 | [0.0781, 0.3344] | 0.0040 | 1.0000 | claim-facing |
| 8192 | +0.2500 | [0.1458, 0.3625] | 0.0013 | 0.9792 | claim-facing |
| 12288 | +0.3000 | [0.2000, 0.4000] | 0.3333 | 1.0000 | low-n |

<p align="center">
  <img src="results/figures/research_grade/fig_rg_03_writer_schedule_estimation.png" alt="Writer schedule estimation plot" width="820">
</p>

## Model sketch

HPM-Lite uses a small hybrid memory stack:

```text
input tokens
   │
   ├─ local path: recent exact token mixing
   ├─ recurrent path: compressed stream state
   ├─ fast-weight path: associative update/read memory
   └─ episodic path: sparse fact retrieval
        ↓
      router
        ↓
   answer distribution
```

The v1 path is a smaller learned write/retrieve memory model. The v2 path adds local mixing, selective recurrent state, fast-weight associative memory, episodic retrieval, and a four-path router. The implementation is deliberately small so that the behavior can be audited seed by seed.

<p align="center">
  <img src="results/figures/research_grade/fig_rg_01_model_task_schematic.png" alt="HPM-Lite model and task schematic" width="820">
</p>

## What is measured

The project tracks memory-native diagnostics, not just final accuracy:

- answer exact accuracy
- answer cross-entropy
- retrieval top-1 / top-k
- true fact written rate
- false write rate
- missed fact rate
- written slots per sample
- retrieval margin
- parameter count
- peak VRAM
- wall time
- examples/sec
- step-level training logs

<p align="center">
  <img src="results/figures/research_grade/fig_rg_04_writer_quality_vs_exact.png" alt="Writer quality versus exact recall" width="820">
</p>

## Research-grade figures

The current figure system is generated by one script:

```bash
python scripts/reset_research_grade_figures.py
```

It writes:

```text
results/figures/research_grade/
results/processed/research_grade/
```

The figure set includes:

| Figure | Purpose |
|---|---|
| `fig_rg_02_exact_claim_forest` | claim-facing exact recall with bootstrap intervals |
| `fig_rg_03_writer_schedule_estimation` | effect-size view of `tf600 - tf200` |
| `fig_rg_04_writer_quality_vs_exact` | relationship between writer quality and answer accuracy |
| `fig_rg_05_retrieval_saturated_failure` | failures when retrieval is already near-perfect |
| `fig_rg_06_training_dynamics_lowess` | raw step logs with LOWESS smoothing |
| `fig_rg_07_seed_distribution_ecdf` | distribution view without binning |
| `fig_rg_08_cost_performance_pareto` | accuracy versus VRAM and wall time |
| `fig_rg_09_metric_heatmap` | compact metric summary |
| `fig_rg_10_exploratory_pairgrid` | exploratory pairwise diagnostics |

An interactive Plotly dashboard is also generated:

```text
results/figures/research_grade/interactive/hpm_v2_long_context_parallel_coordinates.html
```

<p align="center">
  <img src="results/figures/research_grade/fig_rg_05_retrieval_saturated_failure.png" alt="Retrieval-saturated failure analysis" width="820">
</p>

## More diagnostics

<details>
<summary>Cost/accuracy Pareto and metric heatmap (click to expand)</summary>

<br>

Exact recall against peak VRAM and training wall time, split by schedule and hardware:

<p align="center">
  <img src="results/figures/research_grade/fig_rg_08_cost_performance_pareto.png" alt="Cost versus performance Pareto frontier" width="820">
</p>

Exact, writer, and retrieval rates across sequence length and schedule in one view:

<p align="center">
  <img src="results/figures/research_grade/fig_rg_09_metric_heatmap.png" alt="Metric heatmap across sequence length and schedule" width="820">
</p>

A larger exploratory graph/statistics atlas (LLM-memory dashboard, compaction frontier, training-dynamics envelopes, writer phase plane, and a 3D context/schedule surface) is generated separately:

```bash
python scripts/make_advanced_research_atlas.py
```

Outputs land in `results/figures/advanced_atlas/`, with the write-up in [`docs/figures/advanced_research_atlas.md`](docs/figures/advanced_research_atlas.md).

</details>

## Quick start

```bash
git clone https://github.com/felixpatriciorei/HPM-Lite-Memory-Model.git
cd HPM-Lite-Memory-Model
python -m pip install -r requirements.txt
python -m pytest -q
```

Regenerate the current result tables and figures:

```bash
python scripts/reset_research_grade_figures.py
```

Run a small v2 smoke/stress run:

```bash
python -u scripts/run_memory_model.py \
  --models hpm_lite_v2 \
  --seq-len 512 \
  --window 256 \
  --d-model 128 \
  --layers 1 \
  --heads 4 \
  --steps 600 \
  --batch-size 16 \
  --device cuda \
  --memory-null-slot \
  --write-mode learned \
  --learned-writer-teacher-forcing-steps 200 \
  --lambda-writer 0.3 \
  --log-every 50 \
  --save-step-log \
  --record-vram \
  --save-checkpoint false \
  --summary-csv results/raw/hpm_v2_512_seed0.csv \
  --seed 0
```

## Repository layout

```text
hpm_lite/                         model, memory, training, and evaluation code
scripts/run_memory_model.py        main experiment runner
scripts/reset_research_grade_figures.py
scripts/make_research_grade_figures.py
                                  current statistics and figure pipeline
results/raw/                       imported seed-level run summaries
results/processed/research_grade/  canonical processed tables
results/figures/research_grade/    current figure set
docs/research_grade_statistics_methods.md
docs/research_grade_results.md
tests/                             unit and integration tests
```

## Model files

The repository keeps the original compact HPM-Lite implementation and the newer v2 implementation side by side:

- `hpm_lite/model.py` contains the original HPM-Lite model used for the first learned-writer KV-recall experiments.
- `hpm_lite/hpm_v2.py` contains the v2 memory components: selective recurrence, fast-weight memory, episodic retrieval, and routing utilities.
- `hpm_lite/hpm_v2_model.py` wraps the v2 components into a trainable/evaluable model interface.

For figure generation, use the canonical entry point:

```bash
python scripts/make_figures.py
```

Older `make_*_figures.py` entry points have been archived under `scripts/legacy_figures/` to keep the top-level scripts directory focused.

Advanced graph/statistics atlas:

```bash
python scripts/make_advanced_research_atlas.py
```

Outputs:

```text
results/figures/advanced_atlas/
results/processed/advanced_atlas/
docs/figures/advanced_research_atlas.md
```

## Main files

```text
hpm_lite/hpm_v2.py
hpm_lite/hpm_v2_model.py
hpm_lite/train.py
hpm_lite/evaluate.py
hpm_lite/memory.py
scripts/run_memory_model.py
scripts/make_research_grade_figures.py
results/processed/research_grade/hpm_v2_research_grade_run_matrix.csv
results/processed/research_grade/hpm_v2_research_grade_inference_summary.csv
results/processed/research_grade/hpm_v2_research_grade_schedule_effects.csv
```

## Scope and limitations

HPM-Lite is a controlled memory experiment. It does not claim to be a general LLM, to beat modern long-context models, or to prove that synthetic key-value recall transfers directly to real language tasks.

Current limitations:

- 12288-token evidence is promising but under-sampled.
- The project still needs local-baseline stress runs at 4096/8192/12288.
- v2 path ablations are planned but not complete.
- Mixed Kaggle/PC runs are useful sensitivity checks, not the primary claim set.
- Per-example failure logs would make the writer/retrieval diagnosis stronger.

## Roadmap

Near-term:

- add more 12288-token seeds
- run long-context local baselines
- add v2 ablations: no episodic path, no fast-weight path, no selective recurrent path, router disabled
- add per-example failure traces
- separate paper-ready figures from exploratory dashboards

Longer-term:

- natural-language key-value recall
- planted-fact document QA
- multi-hop/entity-state tracking
- memory adapter experiments for small frozen LLMs

## Contributing

This started as a solo research/portfolio project, but issues and pull requests are welcome, particularly around the open items in [Roadmap](#roadmap) (v2 ablations, local baselines, additional seeds).

Before opening a PR:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

A few project conventions worth keeping:

- Any new figure should ship with the script that generates it and the underlying CSV in `results/processed/`, not just the PNG.
- Claims in the README should stay tied to committed result files; speculative numbers belong in `docs/`, not the front page.
- Keep additions to `hpm_lite/` covered by a corresponding test in `tests/`.

For larger changes (new memory path, new task variant), opening an issue first to discuss the approach is preferred over a large unsolicited PR.

## About

HPM-Lite is maintained by [Felix Patricio Rei](https://github.com/felixpatriciorei) as a research and portfolio project exploring explicit, auditable memory mechanisms for long-context recall, structured around three questions: whether a model can store and retrieve long-range facts, whether the readout operator uses retrieved memory correctly, and whether apparent success survives controls such as shuffled values, random keys, no retrieval, and random writes.

The figure and reporting style draws on general open-source documentation guidance, scientific figure-design guidance aimed at objective data display, and reporting checklists that expect clearly described error bars and experimental uncertainty, rather than any single template.

## Citation status

No formal paper release yet. Cite the repository directly if you use the code or figures.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
