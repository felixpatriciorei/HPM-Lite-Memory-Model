# HPM-Lite v2 status and next steps

This overlay adds the first real HPM-Lite v2 building blocks. It does **not** claim
that the repository is now a full 1B--3B LLM competitor. It adds the next missing
architecture pieces in a testable way:

- `BlockwiseSelectiveRecurrentState`: input-conditioned blockwise recurrent path.
- `FastWeightBlockMemory`: differentiable block-level associative memory.
- `HpmV2PathRouter`: four-path router for local/recurrent/fast-weight/episodic state.
- `JepaLiteAuxiliary`: auxiliary latent future-block prediction loss.
- `HpmTextMemory`: first text-facing memory wrapper for frozen LLM experiments.

## Why this order

HPM-Lite v0 proved explicit episodic memory on a synthetic KV task. The next
version needs a stronger recurrent path and a real adapter route before attempting
large-language-model integration.

## What this overlay does not do yet

- It does not pretrain a 1B--3B model from scratch.
- It does not download or run Qwen/Gemma/SmolLM automatically.
- It does not replace the existing `HpmLiteModel` training path yet.
- It does not claim JEPA improves exact recall; JEPA is auxiliary only.

## Next integration task

Wire these modules into the existing training script as a new model type:

```text
--models hpm_lite_v2
```

The first v2 integration should route four paths:

```text
local_state
blockwise_selective_state
fast_weight_state
episodic_state
```

Then compare `hpm_lite` vs `hpm_lite_v2` on the existing 2048 KV benchmark.
