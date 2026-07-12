#!/usr/bin/env python3
"""Local LLM memory benchmark.

This script is intentionally self-contained: stdlib + requests only.
It compares full-context prompting, truncation, keyword RAG, optional embedding RAG,
and compact structured slot memory on synthetic long-document factual recall tasks.

The structured slot memory method is NOT a learned HPM writer. It is a control that
measures what a clean explicit memory interface can do before replacing the writer
with the learned HPM-Lite module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

SCRIPT_VERSION = "llm_memory_benchmark_v1"
DEFAULT_BASE_URL = "http://192.168.18.16:1234/v1"
TASKS = [
    "beginning_fact",
    "middle_final_update",
    "stale_conflict",
    "multi_entity",
    "paraphrase_fact",
]
METHODS = [
    "full_context",
    "truncated_head",
    "truncated_tail",
    "keyword_rag",
    "embedding_rag",
    "structured_slot_memory",
]


@dataclass
class MemorySlot:
    entity: str
    field: str
    value: str
    source: str
    order: int


@dataclass
class BenchmarkItem:
    task: str
    seed: int
    doc: str
    question: str
    gold: str
    notes: str


@dataclass
class LLMResult:
    pred: str
    status: str
    latency_sec: float
    error: str = ""


def split_csv_or_space(values: Optional[Sequence[str]], default: Sequence[str]) -> List[str]:
    if not values:
        return list(default)
    out: List[str] = []
    for item in values:
        for part in str(item).split(','):
            part = part.strip()
            if part:
                out.append(part)
    return out


def normalize_value(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("`", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b[A-Z]{2,4}-\d{3,5}\b", text)
    return m.group(0) if m else text.strip()


def score_prediction(pred: str, gold: str) -> Tuple[int, int, str]:
    extracted = extract_code(pred)
    strict = int(normalize_value(extracted) == normalize_value(gold))
    contains = int(normalize_value(gold) in normalize_value(pred))
    return contains, strict, extracted


def make_noise(seed: int, n: int) -> List[str]:
    topics = [
        "calendar sync", "shipping ledger", "meeting notes", "device inventory",
        "school forms", "garden logs", "recipe archive", "draft checklist",
    ]
    lines = []
    for i in range(n):
        topic = topics[(seed + i) % len(topics)]
        lines.append(
            f"Noise {i:04d}: {topic}; reference={seed}-{i}; this line is irrelevant filler."
        )
    return lines


def code(seed: int, offset: int = 0) -> str:
    return f"RQ-{700 + seed + offset}"


def old_code(seed: int, offset: int = 0) -> str:
    return f"ZZ-{300 + seed + offset}"


def make_item(task: str, seed: int, noise_lines: int) -> BenchmarkItem:
    gold = code(seed)
    lines = make_noise(seed, noise_lines)
    question = "What is Alice's access code? Answer with only the code."

    if task == "beginning_fact":
        lines.insert(3, f"Record: Alice access code = {gold}.")
        notes = "Target fact near beginning, long tail distractor."

    elif task == "middle_final_update":
        lines.insert(5, f"Record: Alice access code = {old_code(seed, 1)}.")
        mid = max(10, len(lines) // 2)
        lines.insert(mid, f"Authoritative update: Alice access code is now {gold}.")
        lines.insert(mid + 4, f"Record: Bob access code = {old_code(seed, 2)}.")
        notes = "Final update in middle, followed by long unrelated tail."

    elif task == "stale_conflict":
        # Many stale values with the same keywords. Naive keyword top-k often retrieves early stale chunks.
        insert_at = 4
        for j in range(12):
            lines.insert(insert_at + j * 3, f"Old record {j}: Alice access code = {old_code(seed, j)}.")
        final_pos = min(len(lines) - 30, max(20, len(lines) // 2 + 25))
        lines.insert(final_pos, f"Final authoritative record: Alice access code = {gold}.")
        notes = "Multiple stale Alice values; only the final authoritative value is correct."

    elif task == "multi_entity":
        entities = ["Bob", "Cara", "Dina", "Eli", "Faye", "Gio", "Hana", "Ivan"]
        pos = 8
        for j, ent in enumerate(entities):
            lines.insert(pos + j * 7, f"Record: {ent} access code = {old_code(seed, j)}.")
        lines.insert(max(12, len(lines) // 2), f"Record: Alice access code = {gold}.")
        notes = "Many entity records; answer requires selecting Alice only."

    elif task == "paraphrase_fact":
        lines.insert(max(8, len(lines) // 3), f"Security note: the login token assigned to Alice is {gold}.")
        question = "What access code should Alice use? Answer with only the code."
        notes = "Question uses access-code wording; document uses login-token wording."

    else:
        raise ValueError(f"unknown task: {task}")

    return BenchmarkItem(task=task, seed=seed, doc="\n".join(lines), question=question, gold=gold, notes=notes)


def get_models(base_url: str, timeout: int) -> List[str]:
    r = requests.get(f"{base_url}/models", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def choose_chat_model(models: Sequence[str]) -> str:
    non_embed = [m for m in models if "embed" not in m.lower()]
    if not non_embed:
        raise RuntimeError("No chat model found. LM Studio is reachable but only embedding models are listed.")
    return non_embed[0]


def choose_embed_model(models: Sequence[str]) -> Optional[str]:
    for m in models:
        if "embed" in m.lower() or "embedding" in m.lower():
            return m
    return None


def classify_error(status_code: Optional[int], text: str, exc: Optional[BaseException] = None) -> Tuple[str, str]:
    raw = text or (repr(exc) if exc is not None else "")
    low = raw.lower()
    if "context length" in low or "n_ctx" in low or "n_keep" in low:
        return "context_limit", raw[:500]
    if isinstance(exc, requests.Timeout):
        return "timeout", raw[:500]
    if status_code is not None:
        return f"http_{status_code}", raw[:500]
    return "error", raw[:500]


def call_llm(base_url: str, model: str, prompt: str, timeout: int, max_tokens: int = 32) -> LLMResult:
    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer with only the requested value. Do not explain."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        latency = time.time() - t0
        if r.status_code >= 400:
            status, err = classify_error(r.status_code, r.text)
            return LLMResult(pred=status.upper(), status=status, latency_sec=-1.0, error=err)
        data = r.json()
        pred = data["choices"][0]["message"]["content"].strip()
        return LLMResult(pred=pred, status="ok", latency_sec=latency)
    except Exception as exc:
        status, err = classify_error(None, "", exc)
        return LLMResult(pred=status.upper(), status=status, latency_sec=-1.0, error=err)


def call_embedding(base_url: str, model: str, text: str, timeout: int) -> Optional[List[float]]:
    try:
        r = requests.post(
            f"{base_url}/embeddings",
            headers={"Content-Type": "application/json"},
            json={"model": model, "input": text},
            timeout=timeout,
        )
        if r.status_code >= 400:
            return None
        data = r.json()
        emb = data.get("data", [{}])[0].get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
    except Exception:
        return None
    return None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def chunk_lines(doc: str, chunk_size: int = 5, stride: int = 3) -> List[str]:
    lines = doc.splitlines()
    chunks = []
    for start in range(0, len(lines), stride):
        chunk = "\n".join(lines[start:start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def keyword_score(question: str, chunk: str) -> float:
    q_words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    c_low = chunk.lower()
    score = 0.0
    for w in q_words:
        if len(w) <= 2:
            continue
        if w in c_low:
            score += 1.0
    # Mild boost for known code/fact markers, but no recency boost.
    for marker in ["alice", "access", "code", "record", "update", "authoritative", "token"]:
        if marker in c_low:
            score += 0.25
    return score


def keyword_retrieve(doc: str, question: str, top_k: int) -> List[str]:
    chunks = chunk_lines(doc)
    scored = [(keyword_score(question, c), i, c) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for s, _i, c in scored[:top_k] if s > 0]


def embedding_retrieve(base_url: str, embed_model: Optional[str], doc: str, question: str, top_k: int, timeout: int) -> Tuple[List[str], str]:
    if not embed_model:
        return [], "embed_unavailable"
    q_emb = call_embedding(base_url, embed_model, question, timeout)
    if q_emb is None:
        return [], "embed_unavailable"
    scored = []
    for i, chunk in enumerate(chunk_lines(doc)):
        emb = call_embedding(base_url, embed_model, chunk, timeout)
        if emb is None:
            return [], "embed_unavailable"
        scored.append((cosine(q_emb, emb), i, chunk))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _s, _i, c in scored[:top_k]], "ok"


def write_structured_slots(doc: str, source: str) -> List[MemorySlot]:
    slots: Dict[Tuple[str, str], MemorySlot] = {}
    # Keep line order so repeated assignments overwrite earlier stale values.
    patterns = [
        re.compile(r"\b(?P<entity>[A-Z][a-z]+) access code\s*=\s*(?P<value>[A-Z]{2,4}-\d{3,5})", re.I),
        re.compile(r"\b(?P<entity>[A-Z][a-z]+)'s access code is now (?P<value>[A-Z]{2,4}-\d{3,5})", re.I),
        re.compile(r"\b(?P<entity>[A-Z][a-z]+) access code is now (?P<value>[A-Z]{2,4}-\d{3,5})", re.I),
        re.compile(r"login token assigned to (?P<entity>[A-Z][a-z]+) is (?P<value>[A-Z]{2,4}-\d{3,5})", re.I),
    ]
    for order, line in enumerate(doc.splitlines()):
        for pat in patterns:
            for m in pat.finditer(line):
                entity = m.group("entity").capitalize()
                value = m.group("value").upper()
                key = (entity.lower(), "access_code")
                slots[key] = MemorySlot(entity=entity, field="access_code", value=value, source=source, order=order)
    return sorted(slots.values(), key=lambda s: (s.entity, s.field))


def retrieve_slots(slots: Sequence[MemorySlot], question: str, top_k: int) -> List[MemorySlot]:
    q = question.lower()
    hits = [s for s in slots if s.entity.lower() in q]
    if not hits:
        hits = list(slots)
    hits.sort(key=lambda s: (0 if s.entity.lower() in q else 1, -s.order))
    return hits[:top_k]


def prompt_full(item: BenchmarkItem) -> str:
    return f"DOCUMENT:\n{item.doc}\n\nQUESTION:\n{item.question}\n"


def prompt_truncated_head(item: BenchmarkItem, char_budget: int) -> str:
    text = item.doc[:char_budget]
    return f"TRUNCATED DOCUMENT HEAD:\n{text}\n\nQUESTION:\n{item.question}\n"


def prompt_truncated_tail(item: BenchmarkItem, char_budget: int) -> str:
    text = item.doc[-char_budget:]
    return f"TRUNCATED DOCUMENT TAIL:\n{text}\n\nQUESTION:\n{item.question}\n"


def prompt_chunks(item: BenchmarkItem, chunks: Sequence[str], label: str) -> str:
    joined = "\n---\n".join(chunks)
    return f"{label}:\n{joined}\n\nQUESTION:\n{item.question}\n"


def prompt_slots(item: BenchmarkItem, slots: Sequence[MemorySlot]) -> str:
    if slots:
        mem = "\n".join(
            f"[slot] entity={s.entity} field={s.field} value={s.value} source={s.source} order={s.order}"
            for s in slots
        )
    else:
        mem = "[no slots retrieved]"
    return f"VERIFIED STRUCTURED MEMORY:\n{mem}\n\nQUESTION:\n{item.question}\n"


def run_method(
    item: BenchmarkItem,
    method: str,
    base_url: str,
    chat_model: str,
    embed_model: Optional[str],
    timeout: int,
    top_k: int,
    trunc_chars: int,
) -> Dict[str, Any]:
    retrieved_items = 0
    retrieval_status = "ok"

    if method == "full_context":
        prompt = prompt_full(item)
    elif method == "truncated_head":
        prompt = prompt_truncated_head(item, trunc_chars)
    elif method == "truncated_tail":
        prompt = prompt_truncated_tail(item, trunc_chars)
    elif method == "keyword_rag":
        chunks = keyword_retrieve(item.doc, item.question, top_k)
        retrieved_items = len(chunks)
        prompt = prompt_chunks(item, chunks, "RETRIEVED CHUNKS")
    elif method == "embedding_rag":
        chunks, retrieval_status = embedding_retrieve(base_url, embed_model, item.doc, item.question, top_k, timeout)
        retrieved_items = len(chunks)
        if retrieval_status != "ok":
            result = LLMResult(pred=retrieval_status.upper(), status=retrieval_status, latency_sec=-1.0, error="embedding endpoint unavailable or failed")
            prompt = ""
        else:
            prompt = prompt_chunks(item, chunks, "EMBEDDING RETRIEVED CHUNKS")
            result = call_llm(base_url, chat_model, prompt, timeout)
    elif method == "structured_slot_memory":
        all_slots = write_structured_slots(item.doc, source=f"{item.task}:{item.seed}")
        slots = retrieve_slots(all_slots, item.question, top_k=1)
        retrieved_items = len(slots)
        prompt = prompt_slots(item, slots)
    else:
        raise ValueError(f"unknown method: {method}")

    if method != "embedding_rag" or retrieval_status == "ok":
        result = call_llm(base_url, chat_model, prompt, timeout)

    exact_contains, strict_value_exact, extracted = score_prediction(result.pred, item.gold)
    return {
        "script_version": SCRIPT_VERSION,
        "task": item.task,
        "seed": item.seed,
        "method": method,
        "gold": item.gold,
        "pred": result.pred,
        "extracted_value": extracted,
        "exact_contains": exact_contains,
        "strict_value_exact": strict_value_exact,
        "status": result.status,
        "latency_sec": round(result.latency_sec, 3) if result.latency_sec >= 0 else -1.0,
        "retrieved_items": retrieved_items,
        "prompt_chars": len(prompt),
        "notes": item.notes,
        "error": result.error,
    }


def mean_or_blank(values: Sequence[float]) -> Any:
    vals = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    if not vals:
        return ""
    return round(sum(vals) / len(vals), 3)


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["task"], r["method"]), []).append(r)
    out = []
    for (task, method), rs in sorted(groups.items()):
        n = len(rs)
        out.append({
            "script_version": SCRIPT_VERSION,
            "task": task,
            "method": method,
            "n": n,
            "exact_contains_mean": round(sum(int(r["exact_contains"]) for r in rs) / n, 4),
            "strict_value_exact_mean": round(sum(int(r["strict_value_exact"]) for r in rs) / n, 4),
            "ok_count": sum(1 for r in rs if r["status"] == "ok"),
            "context_limit_count": sum(1 for r in rs if r["status"] == "context_limit"),
            "embed_unavailable_count": sum(1 for r in rs if r["status"] == "embed_unavailable"),
            "mean_latency_sec": mean_or_blank([float(r["latency_sec"]) for r in rs]),
            "mean_prompt_chars": mean_or_blank([float(r["prompt_chars"]) for r in rs]),
            "mean_retrieved_items": mean_or_blank([float(r["retrieved_items"]) for r in rs]),
        })
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run local LLM memory benchmark.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default="", help="Chat model id. Default: first non-embedding model from /models.")
    p.add_argument("--embed-model", default="", help="Embedding model id. Default: first embedding model from /models.")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--tasks", nargs="*", default=TASKS, help="Space or comma separated task names.")
    p.add_argument("--methods", nargs="*", default=METHODS, help="Space or comma separated method names.")
    p.add_argument("--noise-lines", type=int, default=520)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--trunc-chars", type=int, default=8500)
    p.add_argument("--rows-out", default="results/processed/llm_memory_benchmark_rows.csv")
    p.add_argument("--summary-out", default="results/processed/llm_memory_benchmark_summary.csv")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    tasks = split_csv_or_space(args.tasks, TASKS)
    methods = split_csv_or_space(args.methods, METHODS)

    bad_tasks = [t for t in tasks if t not in TASKS]
    bad_methods = [m for m in methods if m not in METHODS]
    if bad_tasks:
        raise SystemExit(f"Unknown tasks: {bad_tasks}. Valid: {TASKS}")
    if bad_methods:
        raise SystemExit(f"Unknown methods: {bad_methods}. Valid: {METHODS}")

    models = get_models(args.base_url, args.timeout)
    chat_model = args.model or choose_chat_model(models)
    embed_model = args.embed_model or choose_embed_model(models)

    print("SCRIPT_VERSION", SCRIPT_VERSION)
    print("base_url", args.base_url)
    print("chat_model", chat_model)
    print("embed_model", embed_model or "NONE")
    print("tasks", ",".join(tasks))
    print("methods", ",".join(methods))
    print("seeds", args.seeds)

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        for seed in range(args.seeds):
            item = make_item(task, seed, args.noise_lines)
            for method in methods:
                row = run_method(
                    item=item,
                    method=method,
                    base_url=args.base_url,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    timeout=args.timeout,
                    top_k=args.top_k,
                    trunc_chars=args.trunc_chars,
                )
                rows.append(row)
                compact = {k: row[k] for k in [
                    "script_version", "task", "seed", "method", "gold", "pred",
                    "extracted_value", "exact_contains", "strict_value_exact", "status",
                    "latency_sec", "retrieved_items", "prompt_chars",
                ]}
                print(json.dumps(compact, ensure_ascii=False))
                sys.stdout.flush()

    summary = summarize(rows)
    print("SUMMARY")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))

    rows_out = Path(args.rows_out)
    summary_out = Path(args.summary_out)
    write_csv(rows_out, rows)
    write_csv(summary_out, summary)
    print("wrote", rows_out)
    print("wrote", summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
