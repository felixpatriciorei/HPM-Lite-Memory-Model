# Fact-checking and Reproducibility Plan

This project should be judged by reproducible evidence, not by motivation.

## Minimum evidence for a result

Every claimed result should include:

- commit hash
- command
- hardware
- PyTorch version
- CUDA availability
- model parameter count
- sequence length
- local window size
- batch size
- seed
- answer exact accuracy
- retrieval top-1/top-k for memory models
- train wall time
- VRAM peak, when available
- output CSV

## Clean clone reproduction

From a new folder:

```bash
git clone https://github.com/felixpatriciorei/HPM-Lite-Memory-Model.git
cd HPM-Lite-Memory-Model
pip install -r requirements.txt
pytest -q
```

Run a small smoke experiment:

```bash
python scripts/run_memory_model.py   --seq-len 512   --window 256   --d-model 96   --layers 1   --heads 4   --steps 20   --batch-size 4   --device cpu
```

## Leakage checks

A valid memory experiment must rule out shortcuts.

Check:

- answer token is not visible to the model before the answer position
- query key is not paired with the answer in the local window
- local baseline fails when the relevant fact is beyond the local window
- shuffled-value control breaks HPM answer accuracy
- random-key control breaks HPM answer accuracy
- no-retrieval control breaks HPM answer accuracy
- random-write control does not solve the task

## Null-slot checks

Top-k memory retrieval should not be forced to return a bad memory when no good memory exists.

Required tests:

- valid-key queries: null slot should usually be ignored
- missing-key queries: null slot should receive high mass
- distractor-only memory: output should not confidently copy unrelated values
- near-duplicate keys: retrieval margin should be reported

## Ablation checks

Run:

| Variant | Purpose |
|---|---|
| full HPM-Lite | target architecture |
| no episodic | proves exact memory path matters |
| no recurrent | tests continuity path |
| no router | tests learned path weighting |
| local only | baseline |

The strongest evidence is not just that full HPM-Lite works. It is that the right ablation fails for the right reason.

## Reporting rule

Do not hide failed runs. Failed runs should be recorded in a short table:

| Date | Commit | Command | Failure | Fix |
|---|---|---|---|---|

Examples:

- CUDA OOM at seq-len 4096 with batch 32 because attention allocated a large score tensor.
- Fixed by true sliding-window attention and smaller batch.
