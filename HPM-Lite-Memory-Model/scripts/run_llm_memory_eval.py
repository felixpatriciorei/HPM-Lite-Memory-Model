"""Local LLM memory evaluation harness for HPM-Lite.

Compares a local LM Studio chat model against several memory/retrieval conditions:
- full_context: the whole synthetic document is given to the LLM; may fail on context limit.
- truncated_tail: only the end of the document is given.
- keyword_rag: keyword-selected evidence lines are given.
- hpm_symbolic_memory: compact explicit memory slots are given.

This is a systems proof-of-concept harness. The HPM writer here is symbolic, not the
learned writer from the HPM-Lite training benchmark yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

SCRIPT_VERSION = "llm_eval_hardfix_v1"

DEFAULT_BASE_URL = "http://192.168.18.16:1234/v1"
DEFAULT_TASKS = ["beginning_fact", "late_update", "multi_entity"]
DEFAULT_METHODS = ["full_context", "truncated_tail", "keyword_rag", "hpm_symbolic_memory"]


@dataclass
class MemorySlot:
    subject: str
    field: str
    value: str
    source: str
    order: int


@dataclass
class EvalCase:
    task: str
    seed: int
    doc: str
    question: str
    gold: str
    source: str


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(pred: str, gold: str) -> int:
    if pred in {"CTX_LIMIT", "ERROR", "TIMEOUT", "NO_RETRIEVAL"}:
        return 0
    p = normalize_text(pred)
    g = normalize_text(gold)
    return int(p == g or g in p)


def request_json(method: str, url: str, **kwargs):
    r = requests.request(method, url, **kwargs)
    if r.status_code >= 400:
        print("LM Studio error status:", r.status_code)
        print(r.text[:1000])
    r.raise_for_status()
    return r.json()


def pick_chat_model(base_url: str, preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    data = request_json("GET", base_url.rstrip("/") + "/models", timeout=10)
    models = data.get("data", [])
    if not models:
        raise RuntimeError("LM Studio is reachable, but no model is loaded.")
    for m in models:
        mid = m.get("id", "")
        if "embed" not in mid.lower() and "embedding" not in mid.lower():
            return mid
    return models[0].get("id", "local-model")


def call_llm(prompt: str, base_url: str, model: str, timeout_sec: int = 45) -> Tuple[str, float, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    t0 = time.time()
    try:
        data = request_json(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer exactly and concisely. Return only the requested value when possible."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=timeout_sec,
        )
        dt = time.time() - t0
        text = data["choices"][0]["message"]["content"].strip()
        return text, dt, "ok"
    except requests.exceptions.Timeout:
        return "TIMEOUT", -1.0, "timeout"
    except requests.exceptions.HTTPError as e:
        msg = ""
        if getattr(e, "response", None) is not None:
            msg = e.response.text
        if "n_keep" in msg or "context length" in msg or "n_ctx" in msg:
            return "CTX_LIMIT", -1.0, "context_limit"
        return "ERROR", -1.0, "http_error"
    except Exception as e:
        print("call_llm failed:", repr(e))
        return "ERROR", -1.0, "exception"


def noise_lines(seed: int, n: int) -> List[str]:
    out = []
    for i in range(n):
        out.append(
            f"Noise line {i:04d}: filler sentence for seed {seed}; unrelated project notes, ordinary text, and distractor tokens."
        )
    return out


def make_case(task: str, seed: int, n_noise: int = 300) -> EvalCase:
    source = f"synthetic_{task}_{seed}"
    alice_code = f"RQ-{700 + seed}"
    old_code = f"ZX-{200 + seed}"
    bob_office = f"North Annex {seed}"
    cara_project = f"Project Vela {seed}"
    lines = noise_lines(seed, n_noise)

    if task == "beginning_fact":
        lines.insert(5, f"Important fact: Alice's access code is {alice_code}.")
        question = "What is Alice's access code?"
        gold = alice_code
    elif task == "late_update":
        lines.insert(5, f"Important fact: Alice's access code is {old_code}.")
        lines.insert(n_noise - 5, f"Correction update: Alice's access code is now {alice_code}.")
        question = "What is Alice's current access code?"
        gold = alice_code
    elif task == "multi_entity":
        lines.insert(10, f"Important fact: Alice's access code is {alice_code}.")
        lines.insert(120, f"Important fact: Bob's office is {bob_office}.")
        lines.insert(240, f"Important fact: Cara's project is {cara_project}.")
        question = "What is Alice's access code?"
        gold = alice_code
    else:
        raise ValueError(f"unknown task: {task}")

    return EvalCase(task=task, seed=seed, doc="\n".join(lines), question=question, gold=gold, source=source)


def prompt_full_context(case: EvalCase) -> str:
    return f"""Read the document and answer the question.

DOCUMENT:
{case.doc}

QUESTION:
{case.question}

Answer with only the requested value."""


def prompt_truncated_tail(case: EvalCase, keep_lines: int = 80) -> str:
    tail = "\n".join(case.doc.splitlines()[-keep_lines:])
    return f"""Read the visible tail of the document and answer the question. The beginning may be missing.

VISIBLE DOCUMENT TAIL:
{tail}

QUESTION:
{case.question}

Answer with only the requested value. If the answer is not present, say UNKNOWN."""


def keyword_retrieve_lines(case: EvalCase, max_lines: int = 8) -> List[str]:
    q = case.question.lower()
    subjects = ["alice", "bob", "cara"]
    chosen_subjects = [s for s in subjects if s in q] or subjects
    scored = []
    for i, line in enumerate(case.doc.splitlines()):
        low = line.lower()
        score = 0
        for s in chosen_subjects:
            if s in low:
                score += 5
        for word in ["access", "code", "office", "project", "current", "correction", "update", "now"]:
            if word in q and word in low:
                score += 1
            elif word in low:
                score += 0.25
        if score > 0:
            # Prefer later updates when score ties.
            scored.append((score, i, line))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [line for _, _, line in scored[:max_lines]]


def prompt_keyword_rag(case: EvalCase) -> str:
    evidence = keyword_retrieve_lines(case)
    if not evidence:
        evidence = ["NO EVIDENCE RETRIEVED"]
    return f"""Answer using only the retrieved evidence lines.

EVIDENCE:
{chr(10).join(evidence)}

QUESTION:
{case.question}

Answer with only the requested value. For updates, use the latest/current value."""


def hpm_symbolic_write(case: EvalCase) -> List[MemorySlot]:
    slots: List[MemorySlot] = []
    for idx, line in enumerate(case.doc.splitlines()):
        m = re.search(r"Alice's access code is(?: now)? ([A-Z]+-\d+)", line)
        if m:
            slots.append(MemorySlot("Alice", "access_code", m.group(1), case.source, idx))
        m = re.search(r"Bob's office is ([A-Za-z ]+\d+)", line)
        if m:
            slots.append(MemorySlot("Bob", "office", m.group(1).strip(), case.source, idx))
        m = re.search(r"Cara's project is ([A-Za-z ]+\d+)", line)
        if m:
            slots.append(MemorySlot("Cara", "project", m.group(1).strip(), case.source, idx))
    return slots


def hpm_retrieve(case: EvalCase, slots: Sequence[MemorySlot]) -> List[MemorySlot]:
    q = case.question.lower()
    selected = [s for s in slots if s.subject.lower() in q]
    # If multiple slots for same subject/field exist, keep latest by source order.
    latest: Dict[Tuple[str, str], MemorySlot] = {}
    for s in selected:
        key = (s.subject, s.field)
        if key not in latest or s.order > latest[key].order:
            latest[key] = s
    return sorted(latest.values(), key=lambda s: s.order)


def prompt_hpm_memory(case: EvalCase) -> Tuple[str, int]:
    slots = hpm_retrieve(case, hpm_symbolic_write(case))
    if not slots:
        mem = "NO MEMORY RETRIEVED"
    else:
        mem = "\n".join(
            f"[memory] subject={s.subject} field={s.field} value={s.value} source={s.source} order={s.order}"
            for s in slots
        )
    prompt = f"""Use the verified explicit memory slots to answer.

MEMORY:
{mem}

QUESTION:
{case.question}

Answer with only the requested value. For updates, use the latest/current memory value."""
    return prompt, len(slots)


def build_prompt(method: str, case: EvalCase) -> Tuple[str, int]:
    if method == "full_context":
        return prompt_full_context(case), 0
    if method == "truncated_tail":
        return prompt_truncated_tail(case), 0
    if method == "keyword_rag":
        return prompt_keyword_rag(case), len(keyword_retrieve_lines(case))
    if method == "hpm_symbolic_memory":
        return prompt_hpm_memory(case)
    raise ValueError(f"unknown method: {method}")


def run(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    model = pick_chat_model(base_url, args.model)
    tasks = args.tasks
    methods = args.methods

    print("SCRIPT_VERSION", SCRIPT_VERSION)
    print("base_url", base_url)
    print("model", model)
    print("tasks", ",".join(tasks))
    print("methods", ",".join(methods))

    rows = []
    for task in tasks:
        for seed in range(args.seeds):
            case = make_case(task, seed, n_noise=args.noise_lines)
            for method in methods:
                prompt, retrieved = build_prompt(method, case)
                pred, latency, status = call_llm(prompt, base_url, model, timeout_sec=args.timeout)
                row = {
                    "script_version": SCRIPT_VERSION,
                    "task": task,
                    "seed": seed,
                    "method": method,
                    "gold": case.gold,
                    "pred": pred,
                    "exact": exact_match(pred, case.gold),
                    "status": status,
                    "latency_sec": round(latency, 3),
                    "retrieved_items": retrieved,
                    "prompt_chars": len(prompt),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    out_rows = Path(args.rows_out)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for task in tasks:
        for method in methods:
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            if not group:
                continue
            exact_mean = sum(r["exact"] for r in group) / len(group)
            ok_count = sum(1 for r in group if r["status"] == "ok")
            ctx_count = sum(1 for r in group if r["status"] == "context_limit")
            lat = [float(r["latency_sec"]) for r in group if float(r["latency_sec"]) >= 0]
            summary_rows.append({
                "script_version": SCRIPT_VERSION,
                "task": task,
                "method": method,
                "n": len(group),
                "exact_mean": round(exact_mean, 4),
                "ok_count": ok_count,
                "context_limit_count": ctx_count,
                "mean_latency_sec": round(sum(lat) / len(lat), 3) if lat else "",
            })

    out_summary = Path(args.summary_out)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("SUMMARY")
    for r in summary_rows:
        print(json.dumps(r, ensure_ascii=False))
    print("wrote", out_rows)
    print("wrote", out_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local LLM prompting against HPM-style explicit memory.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--noise-lines", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--rows-out", default="results/processed/llm_memory_eval_rows.csv")
    parser.add_argument("--summary-out", default="results/processed/llm_memory_eval_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
