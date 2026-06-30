# LLM memory evaluation prototype

This experiment connects HPM-Lite-style explicit memory to a local LLM served through LM Studio's OpenAI-compatible API.

## Setup

- Backend: LM Studio local server
- Chat model id: `uncategorized`
- Model family: Gemma 7.5B GGUF
- Server: `http://192.168.18.16:1234/v1`
- Script: `scripts/run_llm_memory_eval.py`
- Script version: `llm_eval_hardfix_v1`

## Methods

The current harness compares:

| Method | Description |
|---|---|
| `full_context` | Sends the whole synthetic document to the local LLM. |
| `truncated_tail` | Sends only the tail of the document. |
| `keyword_rag` | Retrieves keyword-matched text chunks and sends them to the LLM. |
| `hpm_symbolic_memory` | Writes compact structured memory slots and sends only retrieved slots to the LLM. |

## Tasks

| Task | Purpose |
|---|---|
| `beginning_fact` | Tests whether a fact near the beginning survives long distractor text. |
| `late_update` | Tests whether the latest update can be used. |
| `multi_entity` | Tests whether the requested entity can be selected among multiple facts. |

## Current result

Run command:

```bash
python scripts/run_llm_memory_eval.py --seeds 5 --tasks beginning_fact late_update multi_entity --base-url http://192.168.18.16:1234/v1