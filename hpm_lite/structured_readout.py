from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .data import CONDITION_RANGE, IF, VALUE_RANGE, VOCAB_SIZE


class CandidateSetReadout(nn.Module):
    """Tiny multi-label readout over retrieved candidate memory values."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, top_scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        k = top_scores.size(1)
        ranks = torch.linspace(0.0, 1.0, steps=k, device=top_scores.device, dtype=top_scores.dtype)
        features = torch.stack(
            [
                top_scores.detach(),
                weights.detach(),
                ranks[None, :].expand_as(top_scores),
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


def controlled_memory_value_tokens(batch: Dict[str, torch.Tensor], memory_control: str) -> torch.Tensor:
    positions = batch["memory_token_positions"][:, :, 1]
    bsz = batch["input_ids"].size(0)
    value_tokens = batch["input_ids"][torch.arange(bsz, device=positions.device)[:, None], positions]
    if memory_control in {"shuffle_values", "shuffled_values"}:
        return torch.roll(value_tokens, shifts=1, dims=1)
    if memory_control == "corrupt_values":
        low, high = VALUE_RANGE
        return low + ((value_tokens - low + 97) % (high - low))
    return value_tokens


def controlled_memory_key_tokens(batch: Dict[str, torch.Tensor], memory_control: str) -> torch.Tensor:
    positions = batch["memory_token_positions"][:, :, 0]
    bsz = batch["input_ids"].size(0)
    key_tokens = batch["input_ids"][torch.arange(bsz, device=positions.device)[:, None], positions]
    if memory_control == "random_keys":
        return torch.full_like(key_tokens, -1)
    return key_tokens


def memory_condition_tokens(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "memory_condition_positions" in batch:
        positions = batch["memory_condition_positions"]
        bsz = batch["input_ids"].size(0)
        safe_positions = positions.clamp_min(0).clamp_max(batch["input_ids"].size(1) - 1)
        tokens = batch["input_ids"][torch.arange(bsz, device=positions.device)[:, None], safe_positions]
        return torch.where(positions >= 0, tokens, torch.full_like(tokens, -1))
    value_positions = batch["memory_token_positions"][:, :, 1]
    condition_positions = value_positions + 2
    bsz = batch["input_ids"].size(0)
    condition_tokens = batch["input_ids"][
        torch.arange(bsz, device=value_positions.device)[:, None],
        condition_positions.clamp_max(batch["input_ids"].size(1) - 1),
    ]
    if_positions = value_positions + 1
    if_tokens = batch["input_ids"][
        torch.arange(bsz, device=value_positions.device)[:, None],
        if_positions.clamp_max(batch["input_ids"].size(1) - 1),
    ]
    return torch.where(if_tokens == IF, condition_tokens, torch.full_like(condition_tokens, -1))


def controlled_memory_condition_tokens(batch: Dict[str, torch.Tensor], memory_control: str) -> torch.Tensor:
    condition_tokens = memory_condition_tokens(batch)
    if memory_control == "corrupt_conditions":
        low, high = CONDITION_RANGE
        valid = condition_tokens >= low
        corrupted = low + ((condition_tokens - low + 17) % (high - low))
        return torch.where(valid, corrupted, condition_tokens)
    return condition_tokens


def query_key_tokens(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    row = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)
    return batch["input_ids"][row, batch["query_key_positions"]]


def query_condition_tokens(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    row = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)
    return batch["input_ids"][row, batch["query_key_positions"] + 1]


def condition_exact_matches(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
    keys = controlled_memory_key_tokens(batch, memory_control)
    conditions = controlled_memory_condition_tokens(batch, memory_control)
    q_keys = query_key_tokens(batch)
    q_conditions = query_condition_tokens(batch)
    return (
        (keys == q_keys[:, None])
        & (conditions == q_conditions[:, None])
        & batch["memory_mask"]
        & (memory_control != "no_retrieval")
    )


def set_slot_labels(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
    if memory_control == "no_retrieval":
        return torch.zeros_like(batch["memory_mask"], dtype=torch.bool)
    keys = controlled_memory_key_tokens(batch, memory_control)
    values = controlled_memory_value_tokens(batch, memory_control)
    q_keys = query_key_tokens(batch)
    answers = batch["answer_token_spans"]
    answer_mask = batch["answer_target_mask"]
    value_is_answer = ((values[:, :, None] == answers[:, None, :]) & answer_mask[:, None, :]).any(dim=-1)
    return (keys == q_keys[:, None]) & value_is_answer & batch["memory_mask"]


def _safe_token_ids(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.clamp_min(0)


def _make_mlp(input_dim: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    modules = []
    current = input_dim
    for _ in range(max(layers - 1, 0)):
        modules.append(nn.Linear(current, hidden))
        modules.append(nn.Tanh())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        current = hidden
    modules.append(nn.Linear(current, 1))
    return nn.Sequential(*modules)


def _set_prf(predicted: torch.Tensor, gold: torch.Tensor) -> Tuple[float, float, float]:
    tp = (predicted & gold).sum().item()
    fp = (predicted & ~gold).sum().item()
    fn = (~predicted & gold).sum().item()
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    return precision, recall, f1


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def symbolic_kv_reader(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, torch.Tensor]:
    if memory_control == "no_retrieval":
        bsz = batch["input_ids"].size(0)
        return {
            "available": torch.zeros(bsz, dtype=torch.bool, device=batch["input_ids"].device),
            "chosen_slots": torch.full((bsz,), -1, dtype=torch.long, device=batch["input_ids"].device),
            "predicted_values": torch.full((bsz,), -1, dtype=torch.long, device=batch["input_ids"].device),
        }
    keys = controlled_memory_key_tokens(batch, memory_control)
    values = controlled_memory_value_tokens(batch, memory_control)
    matches = (keys == query_key_tokens(batch)[:, None]) & batch["memory_mask"]
    scores = matches.long().masked_fill(~batch["memory_mask"], -1)
    chosen = scores.argmax(dim=1)
    available = matches.any(dim=1)
    safe = chosen.clamp_min(0)
    predicted = values.gather(1, safe[:, None]).squeeze(1)
    predicted = torch.where(available, predicted, torch.full_like(predicted, -1))
    chosen = torch.where(available, chosen, torch.full_like(chosen, -1))
    return {"available": available, "chosen_slots": chosen, "predicted_values": predicted}


@torch.no_grad()
def symbolic_kv_metrics(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, float]:
    pred = symbolic_kv_reader(batch, memory_control)
    available = pred["available"]
    if not bool(available.any().item()):
        return {"symbolic_readout_available": 0.0}
    exact = pred["predicted_values"] == batch["answer_tokens"]
    return {
        "symbolic_readout_available": float(available.float().mean().item()),
        "symbolic_kv_exact": float(exact[available].float().mean().item()),
    }


def symbolic_set_reader(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, torch.Tensor]:
    labels = set_slot_labels(batch, memory_control)
    return {
        "available": batch["memory_mask"].any(dim=1) & (memory_control != "no_retrieval"),
        "selected_slots": labels,
        "selected_values": controlled_memory_value_tokens(batch, memory_control),
    }


@torch.no_grad()
def symbolic_set_metrics(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, float]:
    pred = symbolic_set_reader(batch, memory_control)
    available = pred["available"]
    if not bool(available.any().item()):
        return {"symbolic_readout_available": 0.0}
    selected_values = pred["selected_values"]
    selected_slots = pred["selected_slots"]
    gold_slots = set_slot_labels(batch, memory_control)
    answers = batch["answer_token_spans"]
    answer_mask = batch["answer_target_mask"]
    value_matches = (selected_values[:, :, None] == answers[:, None, :]) & answer_mask[:, None, :]
    selected_answer_values = (selected_slots[:, :, None] & value_matches).any(dim=1)
    all_answer_values_selected = (selected_answer_values == answer_mask).all(dim=1)
    no_false_positive_slots = ~(selected_slots & ~gold_slots).any(dim=1)
    exact = all_answer_values_selected & no_false_positive_slots & available
    precision, recall, f1 = _set_prf(selected_slots[available], gold_slots[available])
    return {
        "symbolic_readout_available": float(available.float().mean().item()),
        "symbolic_set_exact": float(exact[available].float().mean().item()),
        "symbolic_set_precision": precision,
        "symbolic_set_recall": recall,
        "symbolic_set_f1": f1,
    }


def symbolic_condition_reader(
    batch: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> Dict[str, torch.Tensor]:
    return symbolic_condition_binding_predictions(batch, memory_control)


def symbolic_longhop_reader(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, torch.Tensor]:
    if memory_control == "no_retrieval":
        bsz = batch["input_ids"].size(0)
        return {
            "available": torch.zeros(bsz, dtype=torch.bool, device=batch["input_ids"].device),
            "predicted_values": torch.full((bsz,), -1, dtype=torch.long, device=batch["input_ids"].device),
        }
    keys = controlled_memory_key_tokens(batch, memory_control)
    values = controlled_memory_value_tokens(batch, memory_control)
    q_keys = query_key_tokens(batch)
    bsz = keys.size(0)
    predicted = torch.full((bsz,), -1, dtype=torch.long, device=keys.device)
    available = torch.zeros((bsz,), dtype=torch.bool, device=keys.device)
    for b in range(bsz):
        valid = batch["memory_mask"][b]
        first = ((keys[b] == q_keys[b]) & valid).nonzero(as_tuple=False).reshape(-1)
        if first.numel() == 0:
            continue
        mid = values[b, first[0]]
        second = ((keys[b] == mid) & valid).nonzero(as_tuple=False).reshape(-1)
        if second.numel() == 0:
            continue
        predicted[b] = values[b, second[0]]
        available[b] = True
    return {"available": available, "predicted_values": predicted}


@torch.no_grad()
def symbolic_longhop_metrics(batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, float]:
    pred = symbolic_longhop_reader(batch, memory_control)
    available = pred["available"]
    if not bool(available.any().item()):
        return {"symbolic_readout_available": 0.0}
    exact = pred["predicted_values"] == batch["answer_tokens"]
    return {
        "symbolic_readout_available": float(available.float().mean().item()),
        "symbolic_longhop_exact": float(exact[available].float().mean().item()),
    }


class LearnedConditionReader(nn.Module):
    """Tiny slot selector for typed conditional memory slots."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        reader_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        train_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, reader_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=reader_dim**-0.5)
        self.embedding.weight.requires_grad_(train_embeddings)
        self.scorer = _make_mlp(reader_dim * 9, hidden, layers, dropout)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(_safe_token_ids(tokens))

    def slot_scores(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
        q_key = self._embed(query_key_tokens(batch))
        q_cond = self._embed(query_condition_tokens(batch))
        key = self._embed(controlled_memory_key_tokens(batch, memory_control))
        cond = self._embed(controlled_memory_condition_tokens(batch, memory_control))
        value = self._embed(controlled_memory_value_tokens(batch, memory_control))
        q_key_expanded = q_key[:, None, :].expand_as(key)
        q_cond_expanded = q_cond[:, None, :].expand_as(cond)
        features = torch.cat(
            [
                q_key_expanded,
                q_cond_expanded,
                key,
                cond,
                value,
                q_key_expanded * key,
                q_cond_expanded * cond,
                torch.abs(q_key_expanded - key),
                torch.abs(q_cond_expanded - cond),
            ],
            dim=-1,
        )
        scores = self.scorer(features).squeeze(-1)
        return scores.masked_fill(~batch["memory_mask"], -1.0e9)

    def loss(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
        if memory_control == "no_retrieval":
            return next(self.parameters()).new_zeros(())
        scores = self.slot_scores(batch, memory_control)
        targets = batch["positive_memory_indices"]
        valid = targets >= 0
        if not bool(valid.any().item()):
            return scores.new_zeros(())
        return F.cross_entropy(scores[valid], targets[valid])

    @torch.no_grad()
    def metrics(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, float]:
        if memory_control == "no_retrieval":
            return {"learned_readout_available": 0.0}
        scores = self.slot_scores(batch, memory_control)
        available = batch["memory_mask"].any(dim=1)
        if not bool(available.any().item()):
            return {"learned_readout_available": 0.0}
        chosen = scores.argmax(dim=1)
        values = controlled_memory_value_tokens(batch, memory_control)
        predicted_values = values.gather(1, chosen[:, None]).squeeze(1)
        exact_matches = condition_exact_matches(batch, memory_control)
        row = torch.arange(batch["input_ids"].size(0), device=batch["input_ids"].device)
        slot_correct = exact_matches[row, chosen] & available
        value_correct = predicted_values == batch["answer_tokens"]
        learned_exact = slot_correct & value_correct
        return {
            "learned_readout_available": float(available.float().mean().item()),
            "learned_condition_exact": float(learned_exact[available].float().mean().item()),
            "learned_condition_slot_accuracy": float(slot_correct[available].float().mean().item()),
            "learned_condition_value_accuracy": float(value_correct[available].float().mean().item()),
        }


class LearnedSetReader(nn.Module):
    """Tiny multi-label reader over structured key/value memory slots."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        reader_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        train_embeddings: bool = True,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.threshold = threshold
        self.embedding = nn.Embedding(vocab_size, reader_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=reader_dim**-0.5)
        self.embedding.weight.requires_grad_(train_embeddings)
        self.scorer = _make_mlp(reader_dim * 5, hidden, layers, dropout)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(_safe_token_ids(tokens))

    def slot_logits(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
        q_key = self._embed(query_key_tokens(batch))
        key = self._embed(controlled_memory_key_tokens(batch, memory_control))
        value = self._embed(controlled_memory_value_tokens(batch, memory_control))
        q_key_expanded = q_key[:, None, :].expand_as(key)
        features = torch.cat(
            [
                q_key_expanded,
                key,
                value,
                q_key_expanded * key,
                torch.abs(q_key_expanded - key),
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1)
        return logits.masked_fill(~batch["memory_mask"], -1.0e9)

    def loss(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> torch.Tensor:
        if memory_control == "no_retrieval":
            return next(self.parameters()).new_zeros(())
        logits = self.slot_logits(batch, memory_control)
        labels = set_slot_labels(batch, memory_control).float()
        mask = batch["memory_mask"]
        if not bool(mask.any().item()):
            return logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])

    @torch.no_grad()
    def metrics(self, batch: Dict[str, torch.Tensor], memory_control: str = "normal") -> Dict[str, float]:
        if memory_control == "no_retrieval":
            return {"learned_readout_available": 0.0}
        logits = self.slot_logits(batch, memory_control)
        available = batch["memory_mask"].any(dim=1)
        if not bool(available.any().item()):
            return {"learned_readout_available": 0.0}
        selected_slots = (torch.sigmoid(logits) >= self.threshold) & batch["memory_mask"]
        selected_values = controlled_memory_value_tokens(batch, memory_control)
        gold_slots = set_slot_labels(batch, memory_control)
        answers = batch["answer_token_spans"]
        answer_mask = batch["answer_target_mask"]
        selected_value_hits = ((selected_values[:, :, None] == answers[:, None, :]) & selected_slots[:, :, None]).any(dim=1)
        all_answer_values_selected = (selected_value_hits == answer_mask).all(dim=1)
        no_false_positive_slots = ~(selected_slots & ~gold_slots).any(dim=1)
        exact = all_answer_values_selected & no_false_positive_slots & available
        precision, recall, f1 = _set_prf(selected_slots[available], gold_slots[available])
        return {
            "learned_readout_available": float(available.float().mean().item()),
            "learned_set_exact": float(exact[available].float().mean().item()),
            "learned_set_precision": precision,
            "learned_set_recall": recall,
            "learned_set_f1": f1,
        }


def _rank_of_mask(scores: torch.Tensor, target_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_scores = scores.masked_fill(~target_mask, -1.0e9)
    target_score = masked_scores.max(dim=1).values
    has_target = target_mask.any(dim=1)
    rank = ((scores > target_score[:, None]) & valid_mask).sum(dim=1).float() + 1.0
    return torch.where(has_target, rank, torch.zeros_like(rank))


@torch.no_grad()
def learned_condition_stress_metrics(
    reader: LearnedConditionReader,
    batch: Dict[str, torch.Tensor],
    memory_control: str = "normal",
    top_k: int = 4,
) -> Dict[str, float]:
    if memory_control == "no_retrieval":
        return {"learned_readout_available": 0.0}
    scores = reader.slot_scores(batch, memory_control)
    mask = batch["memory_mask"]
    available = mask.any(dim=1)
    if not bool(available.any().item()):
        return {"learned_readout_available": 0.0}

    keys = controlled_memory_key_tokens(batch, memory_control)
    conditions = controlled_memory_condition_tokens(batch, memory_control)
    values = controlled_memory_value_tokens(batch, memory_control)
    q_keys = query_key_tokens(batch)
    q_conditions = query_condition_tokens(batch)
    key_match = (keys == q_keys[:, None]) & mask
    cond_match = (conditions == q_conditions[:, None]) & mask
    exact = key_match & cond_match
    key_only = key_match & ~cond_match
    cond_only = cond_match & ~key_match
    neither = ~(key_match | cond_match) & mask
    hard_negative = key_only | cond_only

    chosen = scores.argmax(dim=1)
    row = torch.arange(scores.size(0), device=scores.device)
    top_k = max(1, min(top_k, scores.size(1)))
    top_indices = scores.topk(top_k, dim=1).indices
    topk_correct = exact.gather(1, top_indices).any(dim=1)
    predicted_values = values.gather(1, chosen[:, None]).squeeze(1)
    value_correct = predicted_values == batch["answer_tokens"]

    exact_counts = exact.sum(dim=1)
    correct_rank = _rank_of_mask(scores, exact, mask)
    hard_rank = _rank_of_mask(scores, hard_negative, mask)
    hard_available = hard_negative.any(dim=1) & available
    ambiguous = exact_counts > 1

    denom = available
    return {
        "learned_readout_available": float(available.float().mean().item()),
        "condition_key_match_accuracy": float(key_match[row, chosen][denom].float().mean().item()),
        "condition_cond_match_accuracy": float(cond_match[row, chosen][denom].float().mean().item()),
        "hard_negative_false_positive_rate": float(hard_negative[row, chosen][denom].float().mean().item()),
        "hard_negative_rank_mean": float(hard_rank[hard_available].mean().item()) if bool(hard_available.any().item()) else 0.0,
        "correct_slot_rank": float(correct_rank[denom].mean().item()),
        "learned_top1_slot_accuracy": float(exact[row, chosen][denom].float().mean().item()),
        "learned_topk_slot_accuracy": float(topk_correct[denom].float().mean().item()),
        "learned_value_accuracy": float(value_correct[denom].float().mean().item()),
        "symbolic_available_rate": float((exact_counts > 0)[denom].float().mean().item()),
        "ambiguous_symbolic_match_rate": float(ambiguous[denom].float().mean().item()),
        "key_only_distractor_error_rate": float(key_only[row, chosen][denom].float().mean().item()),
        "condition_only_distractor_error_rate": float(cond_only[row, chosen][denom].float().mean().item()),
        "neither_match_distractor_error_rate": float(neither[row, chosen][denom].float().mean().item()),
    }


def _set_exact_for_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    memory_control: str,
    threshold: float,
) -> torch.Tensor:
    selected = (torch.sigmoid(logits) >= threshold) & batch["memory_mask"]
    values = controlled_memory_value_tokens(batch, memory_control)
    answers = batch["answer_token_spans"]
    answer_mask = batch["answer_target_mask"]
    selected_value_hits = ((values[:, :, None] == answers[:, None, :]) & selected[:, :, None]).any(dim=1)
    all_answer_values_selected = (selected_value_hits == answer_mask).all(dim=1)
    no_false_positive_slots = ~(selected & ~labels).any(dim=1)
    return all_answer_values_selected & no_false_positive_slots


@torch.no_grad()
def learned_set_stress_metrics(
    reader: LearnedSetReader,
    batch: Dict[str, torch.Tensor],
    memory_control: str = "normal",
    top_k: int = 4,
) -> Dict[str, float]:
    if memory_control == "no_retrieval":
        return {"learned_readout_available": 0.0}
    logits = reader.slot_logits(batch, memory_control)
    mask = batch["memory_mask"]
    available = mask.any(dim=1)
    if not bool(available.any().item()):
        return {"learned_readout_available": 0.0}

    labels = set_slot_labels(batch, memory_control)
    selected = (torch.sigmoid(logits) >= reader.threshold) & mask
    row = torch.arange(logits.size(0), device=logits.device)
    chosen = logits.argmax(dim=1)
    top_k = max(1, min(top_k, logits.size(1)))
    top_indices = logits.topk(top_k, dim=1).indices
    topk_correct = labels.gather(1, top_indices).any(dim=1)
    hard_negative = (batch.get("stress_slot_types", torch.zeros_like(mask, dtype=torch.long)) == 2) & mask
    hard_available = hard_negative.any(dim=1) & available
    hard_rank = _rank_of_mask(logits, hard_negative, mask)
    correct_rank = _rank_of_mask(logits, labels, mask)

    missed = (labels & ~selected).sum().float()
    positive_total = labels.sum().float().clamp_min(1.0)
    false_positive = (~labels & selected & mask).sum().float()
    negative_total = (~labels & mask).sum().float().clamp_min(1.0)
    hard_fp = (hard_negative & selected).sum().float()
    hard_total = hard_negative.sum().float().clamp_min(1.0)
    exact_t03 = _set_exact_for_threshold(logits, labels, batch, memory_control, 0.3)
    exact_t07 = _set_exact_for_threshold(logits, labels, batch, memory_control, 0.7)

    return {
        "learned_readout_available": float(available.float().mean().item()),
        "hard_negative_false_positive_rate": float((hard_fp / hard_total).item()),
        "hard_negative_rank_mean": float(hard_rank[hard_available].mean().item()) if bool(hard_available.any().item()) else 0.0,
        "correct_slot_rank": float(correct_rank[available].mean().item()),
        "learned_top1_slot_accuracy": float(labels[row, chosen][available].float().mean().item()),
        "learned_topk_slot_accuracy": float(topk_correct[available].float().mean().item()),
        "missed_positive_rate": float((missed / positive_total).item()),
        "extra_false_positive_rate": float((false_positive / negative_total).item()),
        "learned_set_exact_t03": float(exact_t03[available].float().mean().item()),
        "learned_set_exact_t07": float(exact_t07[available].float().mean().item()),
        "symbolic_available_rate": float(labels.any(dim=1)[available].float().mean().item()),
        "ambiguous_symbolic_match_rate": float((labels.sum(dim=1) > 1)[available].float().mean().item()),
    }


def symbolic_condition_binding_predictions(
    batch: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> Dict[str, torch.Tensor]:
    """Symbolically bind query (key, condition) against pre-query memory slots.

    The prediction path intentionally does not inspect target tokens. Targets are
    only used later by the metric wrapper.
    """

    if memory_control == "no_retrieval":
        bsz = batch["input_ids"].size(0)
        return {
            "available": torch.zeros(bsz, dtype=torch.bool, device=batch["input_ids"].device),
            "chosen_slots": torch.full((bsz,), -1, dtype=torch.long, device=batch["input_ids"].device),
            "predicted_values": torch.full((bsz,), -1, dtype=torch.long, device=batch["input_ids"].device),
            "exact_match_counts": torch.zeros(bsz, dtype=torch.long, device=batch["input_ids"].device),
            "exact_matches": torch.zeros_like(batch["memory_mask"], dtype=torch.bool),
            "chosen_exact": torch.zeros(bsz, dtype=torch.bool, device=batch["input_ids"].device),
        }

    input_ids = batch["input_ids"]
    bsz, slots = batch["memory_mask"].shape
    row = torch.arange(bsz, device=input_ids.device)
    query_key_positions = batch["query_key_positions"]
    query_condition_positions = query_key_positions + 1
    query_keys = input_ids[row, query_key_positions]
    query_conditions = input_ids[row, query_condition_positions]

    memory_keys = controlled_memory_key_tokens(batch, memory_control)
    memory_values = controlled_memory_value_tokens(batch, memory_control)
    memory_conditions = controlled_memory_condition_tokens(batch, memory_control)
    memory_mask = batch["memory_mask"]

    key_match = (memory_keys == query_keys[:, None]) & memory_mask
    condition_match = (memory_conditions == query_conditions[:, None]) & memory_mask
    exact_matches = key_match & condition_match
    scores = key_match.long() + condition_match.long()
    scores = scores.masked_fill(~memory_mask, -1)
    chosen_slots = scores.argmax(dim=1)
    has_any_slot = memory_mask.any(dim=1)
    chosen_slots = torch.where(has_any_slot, chosen_slots, torch.full_like(chosen_slots, -1))
    safe_slots = chosen_slots.clamp_min(0)
    predicted_values = memory_values.gather(1, safe_slots[:, None]).squeeze(1)
    predicted_values = torch.where(has_any_slot, predicted_values, torch.full_like(predicted_values, -1))
    exact_counts = exact_matches.sum(dim=1)
    chosen_exact = exact_matches[row, safe_slots] & has_any_slot

    return {
        "available": has_any_slot,
        "chosen_slots": chosen_slots,
        "predicted_values": predicted_values,
        "exact_match_counts": exact_counts,
        "exact_matches": exact_matches,
        "chosen_exact": chosen_exact,
    }


@torch.no_grad()
def symbolic_condition_binding_metrics(
    batch: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> Dict[str, float]:
    pred = symbolic_condition_binding_predictions(batch, memory_control)
    available = pred["available"]
    if not bool(available.any().item()):
        return {"symbolic_readout_available": 0.0}

    targets = batch["answer_tokens"]
    exact_available = pred["exact_match_counts"] > 0
    ambiguous = pred["exact_match_counts"] > 1
    value_correct = pred["predicted_values"] == targets
    slot_correct = pred["chosen_exact"]
    symbolic_exact = exact_available & slot_correct & value_correct & available
    denom = max(int(available.sum().item()), 1)
    return {
        "symbolic_readout_available": float(available.float().mean().item()),
        "condition_symbolic_exact": float(symbolic_exact[available].float().mean().item()),
        "condition_symbolic_slot_accuracy": float(slot_correct[available].float().mean().item()),
        "condition_symbolic_value_accuracy": float(value_correct[available].float().mean().item()),
        "exact_match_available_rate": float(exact_available[available].float().mean().item()),
        "ambiguous_exact_match_rate": float(ambiguous[available].float().mean().item()),
        "symbolic_binding_hit_1_rate": float(symbolic_exact[available].float().mean().item()),
    }


def candidate_labels(
    batch: Dict[str, torch.Tensor],
    retrieval: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> Tuple[torch.Tensor, torch.Tensor]:
    top_indices = retrieval["top_indices"]
    value_tokens = controlled_memory_value_tokens(batch, memory_control)
    candidate_tokens = value_tokens.gather(1, top_indices)
    answers = batch["answer_token_spans"]
    answer_mask = batch["answer_target_mask"]
    labels = ((candidate_tokens[:, :, None] == answers[:, None, :]) & answer_mask[:, None, :]).any(dim=-1)
    return candidate_tokens, labels.float()


def structured_bce_loss(
    readout: CandidateSetReadout,
    batch: Dict[str, torch.Tensor],
    retrieval: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> torch.Tensor:
    if not retrieval or "top_indices" not in retrieval:
        return torch.zeros((), device=batch["input_ids"].device)
    _, labels = candidate_labels(batch, retrieval, memory_control)
    top_scores = retrieval["scores"].gather(1, retrieval["top_indices"])
    logits = readout(top_scores, retrieval["weights"])
    return F.binary_cross_entropy_with_logits(logits, labels)


@torch.no_grad()
def structured_set_metrics(
    readout: CandidateSetReadout,
    batch: Dict[str, torch.Tensor],
    retrieval: Dict[str, torch.Tensor],
    memory_control: str = "normal",
) -> Dict[str, float]:
    if not retrieval or "top_indices" not in retrieval:
        return {
            "structured_set_exact": 0.0,
            "structured_per_value_f1": 0.0,
            "structured_bce": 0.0,
        }

    _, labels = candidate_labels(batch, retrieval, memory_control)
    top_scores = retrieval["scores"].gather(1, retrieval["top_indices"])
    logits = readout(top_scores, retrieval["weights"])
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    predictions = torch.sigmoid(logits) >= 0.5
    gold = labels.bool()

    exact = (predictions == gold).all(dim=1).float().mean().item()
    tp = (predictions & gold).sum().item()
    fp = (predictions & ~gold).sum().item()
    fn = (~predictions & gold).sum().item()
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    return {
        "structured_set_exact": exact,
        "structured_per_value_f1": f1,
        "structured_bce": float(loss.item()),
    }
