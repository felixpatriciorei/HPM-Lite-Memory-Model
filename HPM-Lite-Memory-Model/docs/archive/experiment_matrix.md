# HPM-Lite Experiment Matrix

## Phase 0 — Safety checkpoint

Before new experiments:
- commit current code
- run tests
- save current README and figures
- do not delete historical data

## Phase 1 — Logging upgrade smoke test

Goal:
Add `--models`, `--log-every`, VRAM logging, and step CSV logging.

Run:

```bash
python scripts/run_memory_model.py --models hpm_lite --seq-len 512 --window 256 --d-model 96 --layers 1 --heads 4 --steps 100 --batch-size 8 --device cuda --memory-null-slot --write-mode learned --learned-writer-teacher-forcing-steps 50 --lambda-writer 0.3 --log-every 25 --save-step-log
```

Win condition:
- prints every 25 steps
- creates run_summary.csv
- creates step_log.csv
- tests still pass

## Phase 2 — Learned writer core sweep

Goal:
512 and 2048 with 3 seeds.

Runs:
- seq_len: 512, 2048
- seeds: 0, 1, 2
- model: hpm_lite
- write_mode: learned

Use RTX 4060 settings:
- 512: batch 16, steps 600
- 2048: batch 8, steps 600

Win condition:
- exact >= 0.95 for HPM-Lite
- writer recall >= 0.98
- retrieval top1 >= 0.98

## Phase 3 — Local baseline sweep

Goal:
Make local comparison fair and logged.

Runs:
- seq_len: 512, 2048, 4096, 8192
- seeds: 0, 1, 2
- model: local
- window: 256

Win condition:
- local fails outside window
- metrics logged consistently

## Phase 4 — Oracle/null-slot long-distance sweep

Goal:
Reconfirm 4096 and 8192 with current logging.

Runs:
- seq_len: 4096, 8192
- seeds: 0, 1, 2 if time allows
- model: hpm_lite
- write_mode: oracle or current null-slot setting

Win condition:
- retrieval top1 near 1
- exact near 1
- cost metrics available

## Phase 5 — Ablations

Variants:
- full_hpm_lite
- no_episodic
- no_recurrent
- no_router
- no_null_slot
- random_write
- shuffled_values
- no_retrieval

Distances:
- 512
- 2048 first
- then 4096 if time allows

Win condition:
- full works
- no episodic fails on long-range recall
- shuffled/random/no-retrieval fail

## Phase 6 — Mechanism traces

Collect on small eval batches:
- retrieval_trace.csv
- router_trace.csv
- memory_graph_nodes.csv
- memory_graph_edges.csv

Use seq_len 512 first.

Win condition:
- can create retrieval score distribution
- can create router/path usage plot
- can create memory interaction graph

## Phase 7 — README update

Only after Phases 1–5:
- replace simple figures with multipanel/heatmap/diagnostic figures
- move old simple figures to archive
- keep limitations clear
