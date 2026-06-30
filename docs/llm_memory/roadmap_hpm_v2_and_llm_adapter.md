# HPM-Lite v2 and HPM-LLM adapter roadmap

This project currently proves a narrow but useful claim: explicit learned episodic memory can beat a fixed-window local Transformer on a controlled long-range key-value recall task.

The next goal is not to train a 1B+ language model from scratch. The next goal is to make HPM-Lite into a memory system that can attach to a small open LLM, then evaluate whether it improves long-range exact recall under fixed context and compute limits.

## Milestone A: logging/schema cleanup

Required before more experiments:

- Record the user-requested `seed` without hidden model offsets.
- Record `d_model`, `layers`, and `heads` in every summary row.
- Mark local baseline `write_mode` as `none` in public summaries.
- Blank local writer/retrieval fields that are not meaningful for the no-memory baseline.

## Milestone B: HPM-Lite v2 blockwise memory

Upgrade the toy token-level HPM into a blockwise model:

- Process local windows normally.
- Summarize chunks of 64/128/256 tokens.
- Update a recurrent memory only at chunk boundaries.
- Write episodic memory slots only when a writer confidence threshold is met.
- Add top-k retrieval aggregation and memory replacement/aging.

The first recurrent upgrade should be input-conditioned, not a full custom CUDA Mamba implementation:

```text
u_b = chunk_summary(hidden[b])
g_b = sigmoid(W_g u_b)
d_b = tanh(W_d u_b)
r_b = g_b * r_{b-1} + (1 - g_b) * d_b
```

This gives us the core selective-state idea in pure PyTorch before attempting high-performance state-space kernels.

## Milestone C: JEPA-lite auxiliary path

JEPA should be auxiliary, not a fact store.

Use it for latent prediction:

```text
context chunks -> predictor -> future latent representation
target encoder -> stop-gradient target latent
loss = representation distance
```

Diagnostics required:

- latent variance
- predictor/target cosine similarity
- collapse score
- downstream exact recall with and without JEPA loss

If JEPA hurts exact recall, keep it disabled for exact-memory benchmarks.

## Milestone D: HPM-LLM memory wrapper

First functional system:

1. Load a frozen 1B-2B LLM.
2. Feed a long document or conversation into an HPM writer.
3. Store facts/events/entities in HPM episodic memory.
4. On a user question, retrieve relevant memory.
5. Insert retrieved memories into the LLM prompt.
6. Compare against:
   - LLM alone
   - LLM + naive keyword retrieval
   - LLM + HPM retrieval

This is the first fair way to compare HPM-Lite to 1B-3B models without pretending we can pretrain a billion-parameter model from scratch.

## Milestone E: soft-prompt memory adapter

After the wrapper works:

- Convert retrieved HPM memory vectors into learned soft prompt tokens.
- Freeze the base LLM.
- Train only the memory writer/retriever/projection adapter.
- Optional: add LoRA after the adapter baseline is stable.

## What not to claim yet

Do not claim:

- HPM-Lite is a general LLM.
- HPM-Lite beats Qwen/Gemma/TinyLlama generally.
- JEPA/Mamba/episodic memory are fully implemented just because the roadmap names them.

Valid near-term claim:

> A learned HPM-style episodic memory module can improve long-range exact recall for small models under fixed local-context limits.
