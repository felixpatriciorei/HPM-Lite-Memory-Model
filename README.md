# HPM-Lite

HPM-Lite is a compact PyTorch research prototype for testing one narrow idea:

> Can a tiny local-attention model with simple episodic memory recover long-range exact facts better than a same-size local-attention baseline?

This repository is meant to be a clean, reproducible research prototype. It focuses on controlled synthetic memory tasks, diagnostic baselines, and failure-mode analysis rather than claiming to be a practical Transformer replacement.

## Highlights

- Local-attention baseline, episodic-memory baseline, HPM-Lite variant, recurrent summary variant, and Hebbian-style associative memory baseline.
- Synthetic key-value, coexisting, conditional, and long-hop style diagnostic tasks.
- Validation controls for no retrieval, shuffled values, random keys, random writes, and leak checks.
- Structured readout experiments for typed memory slots and learned slot extraction.
- CPU-friendly smoke tests and small validation runs.

## Repository Structure

```text
hpm_lite/      Core datasets, models, memory modules, metrics, and read/write utilities
scripts/       Experiment runners and diagnostic scripts
tests/         Pytest tests for shapes, data generation, memory, write modes, and readouts
docs/          Result summaries and structured-memory writeups
```

Generated artifacts such as checkpoints, raw run folders, logs, and `__pycache__` files are intentionally excluded from this GitHub-ready version.

## Install

```bash
pip install -r requirements.txt
```

The code uses synthetic data and does not require an external dataset.

## Smoke Test

```bash
python scripts/run_smoke.py
```

This runs a short CPU sanity check for the core model variants and verifies that losses, exact accuracy, and retrieval metrics run without NaNs.

## Run Validation Controls

```bash
python scripts/run_validation.py --steps 5 --batch-size 4 --d-model 64 --layers 1 --heads 4 --device cpu --eval-batches 2 --write-modes oracle,fact_token,random_write --task kv
```

This evaluates memory behavior under normal retrieval and controls such as no retrieval, shuffled values, and random keys.


## HPM-Lite Memory Model Brief

For the exact two-model comparison in the brief, run:

```bash
python scripts/run_memory_model.py --seq-len 512 --window 256 --d-model 192 --layers 2 --heads 4 --steps 200 --device cpu
```

This trains only:

- `local`: a tiny local-window Transformer baseline.
- `hpm_lite`: local mixer + GRU recurrent state + episodic retrieval + learned softmax router.

The `hpm_lite` forward path implements:

```text
x_t -> embedding
l_t = local mixer
r_t = GRU recurrent state
e_t = episodic memory retrieval
alpha = softmax(W[l_t, r_t, e_t])
m_t = alpha_l l_t + alpha_r r_t + alpha_e e_t
p(y) = softmax(W_o m_t)
```

No JEPA objective, no ANN index, no full global attention, and no large-language-model components are used.

## Structured Memory Diagnostics

```bash
python scripts/run_structured_readers.py
python scripts/run_noisy_slot_extraction.py
```

These scripts test whether typed readouts and learned slot extraction can recover answers when ordinary next-token decoding is not the right read/use operator.

## Tests

```bash
pytest
```

## Interpreting Results

The primary metric is exact answer accuracy at the answer position. Cross-entropy is useful for diagnostics, but small CE shifts are not meaningful unless exact recall and controls also improve.

A memory mechanism is interesting only if it clearly beats the local baseline on long-gap recall while passing controls that rule out leakage or shortcut behavior.

## Limitations

- The tasks are synthetic and controlled.
- Some experiments use oracle or parser-based memory writes to isolate retrieval/readout behavior.
- The project does not test real natural-language extraction, large-vocabulary generation, production inference, or large-scale training.
- HPM-Lite is a research prototype, not a production model architecture.

## Selected Results

See:

- [`docs/results.md`](docs/results.md)
- [`docs/structured_memory_readout.md`](docs/structured_memory_readout.md)
