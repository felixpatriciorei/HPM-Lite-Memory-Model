# Local LLM memory benchmark

This benchmark compares a local LLM served through LM Studio against several memory/retrieval interfaces on long synthetic documents.

This is an evaluation harness, not a final HPM claim. The `structured_slot_memory` method is a clean explicit-memory control. It should be replaced by the learned HPM-Lite writer/retriever in the next integration step.

## Methods

- `full_context`: send the full document to the local LLM.
- `truncated_head`: send only the beginning of the document.
- `truncated_tail`: send only the end of the document.
- `keyword_rag`: retrieve keyword-matched chunks.
- `embedding_rag`: retrieve chunks using the LM Studio embedding endpoint.
- `structured_slot_memory`: parse facts into compact memory slots and retrieve one slot for the target entity.

## Tasks

- `beginning_fact`: target fact appears near the beginning.
- `middle_final_update`: stale value appears early; correct update appears in the middle; unrelated tail follows.
- `stale_conflict`: many stale values share the same keywords; one final authoritative value is correct.
- `multi_entity`: many entities have similar records; the query asks for Alice.
- `paraphrase_fact`: document uses login-token wording while the question asks for access code.

## Example command

```cmd
python scripts\run_llm_memory_benchmark.py --seeds 5 --tasks beginning_fact middle_final_update stale_conflict multi_entity paraphrase_fact --base-url http://192.168.18.16:1234/v1
```

The script also accepts comma-separated task lists:

```cmd
python scripts\run_llm_memory_benchmark.py --seeds 5 --tasks beginning_fact,middle_final_update,stale_conflict,multi_entity,paraphrase_fact --base-url http://192.168.18.16:1234/v1
```

Outputs:

- `results/processed/llm_memory_benchmark_rows.csv`
- `results/processed/llm_memory_benchmark_summary.csv`
