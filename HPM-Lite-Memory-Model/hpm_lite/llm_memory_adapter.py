from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence


FACT_RE = re.compile(r"\bFACT\s+([^\s:]+)\s+([^\s:]+)", re.IGNORECASE)
QUERY_RE = re.compile(r"\bQUERY\s+([^\s:]+)", re.IGNORECASE)


@dataclass(frozen=True)
class TextMemorySlot:
    key: str
    value: str
    source: str = ""


class HpmTextMemory:
    """Tiny text-facing memory adapter for frozen LLM experiments.

    This is not the final neural adapter. It is the first functional bridge:
    extract/store facts, retrieve relevant facts, and inject them into a prompt
    for a frozen 1B--3B LLM.
    """

    def __init__(self, slots: Sequence[TextMemorySlot] | None = None):
        self.slots: List[TextMemorySlot] = list(slots or [])

    def write(self, key: str, value: str, source: str = "") -> None:
        self.slots.append(TextMemorySlot(str(key), str(value), str(source)))

    def ingest_fact_syntax(self, text: str, source: str = "") -> int:
        count = 0
        for key, value in FACT_RE.findall(text):
            self.write(key, value, source=source)
            count += 1
        return count

    @staticmethod
    def _score(query: str, slot: TextMemorySlot) -> float:
        q = query.lower()
        key = slot.key.lower()
        value = slot.value.lower()
        score = 0.0
        if key == q:
            score += 10.0
        if key in q or q in key:
            score += 4.0
        q_terms = set(re.findall(r"[a-zA-Z0-9_]+", q))
        slot_terms = set(re.findall(r"[a-zA-Z0-9_]+", key + " " + value))
        score += len(q_terms & slot_terms)
        return score

    def retrieve(self, query: str, top_k: int = 4) -> List[TextMemorySlot]:
        ranked = sorted(self.slots, key=lambda slot: self._score(query, slot), reverse=True)
        return [slot for slot in ranked[: max(0, int(top_k))] if self._score(query, slot) > 0]

    def answer_fact_query(self, query_text: str) -> str | None:
        match = QUERY_RE.search(query_text)
        key = match.group(1) if match else query_text.strip()
        hits = self.retrieve(key, top_k=1)
        return hits[0].value if hits else None


def build_memory_augmented_prompt(question: str, memories: Iterable[TextMemorySlot]) -> str:
    lines = ["Relevant memory:"]
    any_memory = False
    for slot in memories:
        any_memory = True
        lines.append(f"- {slot.key}: {slot.value}")
    if not any_memory:
        lines.append("- <none>")
    lines.extend(["", "Question:", question, "", "Answer using the relevant memory when it is helpful."])
    return "\n".join(lines)
