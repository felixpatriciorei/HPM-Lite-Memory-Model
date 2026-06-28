# HPM-Lite Memory Model

> A small PyTorch research prototype for testing whether explicit episodic memory helps with long-range exact recall when a local Transformer cannot see the original fact.

This repository is a controlled memory experiment, not a chatbot and not a production LLM. The narrow question is:

> If a key-value fact appears far outside the local attention window, can a small HPM-style model write it into memory and retrieve it later better than a fixed-window local Transformer baseline?

Current evidence says **yes on synthetic key-value recall**. The strongest current result is a 2048-token learned-writer sweep where HPM-Lite reaches **98.33% mean exact answer accuracy over 3 seeds**, while the matched local-window baseline reaches **0.00% over 3 seeds**.

---

## Why this repo exists

Transformers are good at many things, but local attention has an obvious weakness: if the needed token is outside the window, the model cannot directly attend to it. This project isolates that failure mode with a deliberately simple task.

The point is not to prove general intelligence. The point is to make one mechanism testable:

- local context handles nearby tokens,
- recurrent state carries a compressed stream state,
- episodic memory stores sparse key-value facts,
- a router mixes local, recurrent, and episodic paths before prediction.

The experiment is intentionally small enough to run on consumer GPUs, but structured enough to record exact accuracy, answer cross-entropy, retrieval quality, writer quality, parameters, speed, VRAM, and wall time.

---

## Task

The synthetic sequence looks like this:

```text
FACT k12 v77
FACT k03 v19
FACT k88 v41
NOISE ...
QUERY k03
ANSWER v19
```

The model must output the correct value token at the answer position. The difficulty is distance: the relevant `FACT` can appear hundreds or thousands of tokens before the `QUERY`.

In the main 2048-token setting:

```text
sequence length = 2048
local window    = 256
```

So the local baseline cannot directly inspect many of the earlier facts at answer time.

---

## Model sketch

HPM-Lite has three paths:

1. **Local path** — handles short-range token mixing.
2. **Recurrent path** — keeps a compressed sequential state.
3. **Episodic path** — writes and retrieves sparse key-value memories.

The router learns how much to trust each path:

```math
l_t = \mathrm{LocalMixer}(x_{1:t})
```

```math
r_t = \mathrm{GRU}(x_t, r_{t-1})
```

```math
e_t = \mathrm{EpisodicRead}(\kappa_t, M)
```

```math
\alpha = \mathrm{softmax}(W[l_t, r_t, e_t])
```

```math
m_t = \alpha_l l_t + \alpha_r r_t + \alpha_e e_t
```

```math
p(y_t) = \mathrm{softmax}(W_o m_t)
```

In plain language: the model can answer from nearby context, from recurrent state, or from retrieved memory.

---

## Main result: learned writer at 2048 tokens

These are processed seed sweeps, not a single cherry-picked run.

| Model | Seq len | Window | Seeds used | Params | Exact accuracy | Answer CE |
|---|---:|---:|---:|---:|---:|---:|
| HPM-Lite, learned writer | 2048 | 256 | 3 | 721,671 | **0.9833 ± 0.0144** | **0.4943 ± 0.6340** |
| Local Transformer baseline | 2048 | 256 | 3 matched | 522,242 | **0.0000 ± 0.0000** | **6.8873 ± 0.3920** |
| Local Transformer baseline | 2048 | 256 | 4 total | 522,242 | 0.0031 ± 0.0063 | 6.9785 ± 0.3684 |

Seed-level 2048 learned-writer results:

| Seed | HPM exact | HPM CE | Retrieval top1 | True fact written | Missed fact | False write |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.9969 | 0.0031 | 0.0031 |
| 1 | 0.9750 | 0.2737 | 1.0000 | 0.9813 | 0.0188 | 0.0188 |
| 2 | 0.9750 | 1.2091 | 1.0000 | 0.9844 | 0.0156 | 0.0156 |

Seed-level local baseline results:

| Seed | Local exact | Local CE |
|---:|---:|---:|
| 0 | 0.0000 | 6.5883 |
| 1 | 0.0000 | 6.7425 |
| 2 | 0.0000 | 7.3310 |
| 3, extra | 0.0125 | 7.2522 |

**Interpretation:** retrieval top-1 is 100% for HPM-Lite across the 2048 learned-writer seeds. The remaining HPM errors are mostly associated with learned write misses, not failed memory lookup.

**Important caveat:** this is not yet a parameter-matched proof. The HPM model has more parameters than the local baseline in this sweep. The result is still useful because the gap is large, but a stricter parameter-matched control belongs in the next round.

---

## What this supports

The current evidence supports this limited claim:

> On a controlled synthetic long-range key-value recall benchmark, explicit episodic memory lets a small HPM-style model retain and retrieve facts that a fixed-window local Transformer baseline fails to recover at 2048 tokens.

It does **not** prove:

- general language understanding,
- chatbot ability,
- replacement of full attention,
- natural-language fact extraction,
- unsupervised memory writing,
- production readiness,
- superiority to all Transformer variants.

This repo is a mechanism study, not a general AI system.

---

## Repository structure

```text
hpm_lite/
  data.py                 synthetic key-value task generation
  evaluate.py             evaluation and metric computation
  memory.py               episodic memory logic
  metrics.py              accuracy / retrieval metrics
  model.py                local baseline and HPM-Lite model
  train.py                training loop and logging
  write_modes.py          oracle, random, and learned write modes

scripts/
  run_memory_model.py     main experiment runner
  make_research_figures.py
  make_paper_figures.py

docs/
  figure_design_audit.md
  figure_reference_notes.md
  experiment_matrix.md
  logging_schema.md

results/
  processed/              committed processed CSV sweeps
  figures/paper/          generated paper-style figures

tests/
  test_memory.py
  test_learned_writer.py
  test_hpm_lite_router.py
  test_shapes.py
  test_experiment_logging.py
```

---

## Install

```bash
git clone https://github.com/felixpatriciorei/HPM-Lite-Memory-Model.git
cd HPM-Lite-Memory-Model
python -m pip install -r requirements.txt
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Run tests:

```bash
python -m pytest -q
```

---

## Reproduce the 2048 learned-writer run

One HPM seed:

```bash
python -u scripts/run_memory_model.py \
  --models hpm_lite \
  --seq-len 2048 \
  --window 256 \
  --d-model 128 \
  --layers 1 \
  --heads 4 \
  --steps 600 \
  --batch-size 8 \
  --device cuda \
  --memory-null-slot \
  --write-mode learned \
  --learned-writer-teacher-forcing-steps 200 \
  --lambda-writer 0.3 \
  --log-every 100 \
  --save-step-log \
  --record-vram \
  --seed 0
```

One local baseline seed:

```bash
python -u scripts/run_memory_model.py \
  --models local \
  --seq-len 2048 \
  --window 256 \
  --d-model 128 \
  --layers 1 \
  --heads 4 \
  --steps 600 \
  --batch-size 8 \
  --device cuda \
  --log-every 100 \
  --save-step-log \
  --record-vram \
  --seed 0
```

---

## Generate figures

```bash
python scripts/make_research_figures.py
```

Expected output:

```text
results/figures/paper/
  fig_01_model_task_schematic.png
  fig_02_main_2048_results.png
  fig_03_writer_retrieval_diagnostics.png
  fig_04_hpm_training_dynamics.png
  fig_05_supplemental_seed_checks.png
  paper_results_table.csv
  figure_manifest.csv
  figure_audit_report.md
```

The figure pipeline is designed to show raw seed points instead of hiding everything behind a single bar. The audit report also records what the figures should and should not be used to claim.

---

## Data files to trust first

Use processed files before raw run folders:

```text
results/processed/learned_writer_2048_seed_sweep.csv
results/processed/local_2048_seed_sweep.csv
results/processed/learned_writer_512_seed_sweep.csv
```

Known logging caveats:

- Some raw summaries may have blank seed/model-shape fields; the processed sweep files fix the seed assignment.
- For `model=local`, writer-related columns are bookkeeping artifacts and should not be interpreted as local-memory behavior.
- The 2048 HPM-vs-local comparison is strong, but not perfectly parameter-matched.

---

## Roadmap

High-priority next experiments:

- parameter-matched local baseline,
- no-episodic-memory ablation,
- no-recurrent-path ablation,
- no-router or fixed-router ablation,
- shuffled-memory control,
- missing-key/null-slot control,
- 4096-token learned-writer sweep,
- natural-ish fact templates after synthetic KV is stable.

The main standard is simple: if a figure or table makes a claim, there should be a command and a processed CSV behind it.

---

## Citation

```bibtex
@software{hpm_lite_memory_model,
  title = {HPM-Lite Memory Model},
  author = {Felix Patricio},
  year = {2026},
  url = {https://github.com/felixpatriciorei/HPM-Lite-Memory-Model}
}
```

---

## Status

Active research prototype. Promising mechanism evidence; not a finished architecture proof.
