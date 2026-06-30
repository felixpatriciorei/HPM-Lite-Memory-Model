# HPM-Lite Memory Model Results

This document records the current proof object for the HPM-Lite memory model experiment.

## Claim under test

> A small HPM-style model can remember more long-range key-value facts than a local Transformer under similar compute.

The current evidence is limited to synthetic key-value recall with oracle fact writes. It is not a claim about chatbot ability or general long-context language understanding.

## Current table

| Seq len | Window | Local exact | HPM-Lite exact | Gain | Local params | HPM params | Local train sec | HPM train sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 256 | 0.0063 | 1.0000 | +0.9938 | 1,375,490 | 1,784,453 | 20.47 | 25.51 |
| 2048 | 256 | 0.0000 | 1.0000 | +1.0000 | 1,375,682 | 1,784,645 | 1,807.67 | 2,481.45 |
| 4096 | 256 | 0.0000 | 1.0000 | +1.0000 | 784,386 | 966,918 | 1,837.20 | 1,749.30 |
| 8192 | 256 | 0.0000 | 1.0000 | +1.0000 | 944,642 | 1,047,750 | 79.57 | 80.49 |

## Graphs

![Exact recall vs distance](figures/exact_recall_vs_distance.png)

![Exact gain by distance](figures/exact_gain_by_distance.png)

![Answer CE vs distance](figures/answer_ce_vs_distance.png)

## Interpretation

The current runs support a narrow claim: when key-value facts are written into an episodic store, HPM-Lite can retrieve facts far beyond the local attention window, while the local baseline fails.

This is expected but still useful. It verifies that the code path, dataset, retrieval metrics, and answer readout can express the intended memory advantage.

## Important caveats

- Current results are mostly single-seed.
- Some distances use different model sizes due to 8GB VRAM limits.
- Current writes are oracle/parser-style, not fully learned.
- The task is synthetic.
- The results do not yet prove robustness to invalid queries or noisy memory.
- The results do not prove general natural-language conversation.

## Next required evidence

1. Same distance sweep over seeds 0, 1, and 2.
2. Parameter-matched runs.
3. Ablations:
   - full HPM-Lite
   - no episodic
   - no recurrent
   - no router
   - local only
4. Null-slot/no-match controls.
5. VRAM and tokens/sec logging.
6. Report failed runs and OOM limits.
