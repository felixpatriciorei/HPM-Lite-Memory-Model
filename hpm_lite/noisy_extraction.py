from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .data import CONDITION_RANGE, KEY_RANGE, VALUE_RANGE, VOCAB_SIZE
from .write_modes import clone_batch, first_positions
from .data import QUERY


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _shift_with_zero(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    out = torch.zeros_like(x)
    if shift > 0:
        out[:, shift:] = x[:, :-shift]
    else:
        out[:, :shift] = x[:, -shift:]
    return out


class LearnedTypedExtractor(nn.Module):
    """Small pointer extractor for typed synthetic memory slots.

    V1 assumes the number of slots is known and trains against canonical
    occurrence order. That avoids Hungarian matching for now and is reported as
    a limitation in the runner output.
    """

    def __init__(
        self,
        max_slots: int,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.seq_len = seq_len
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.token_emb = nn.Embedding(VOCAB_SIZE, extractor_dim)
        self.pos_emb = nn.Embedding(seq_len, extractor_dim)
        self.slot_emb = nn.Embedding(max_slots, extractor_dim)
        self.field_emb = nn.Embedding(3, extractor_dim)
        modules = []
        in_dim = extractor_dim * 5
        current = in_dim
        for _ in range(max(layers - 1, 0)):
            modules.append(nn.Linear(current, hidden))
            modules.append(nn.Tanh())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            current = hidden
        modules.append(nn.Linear(current, extractor_dim))
        self.context = nn.Sequential(*modules)
        self.scale = extractor_dim ** -0.5

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        windows = [_shift_with_zero(x, shift) for shift in (-2, -1, 0, 1, 2)]
        return self.context(torch.cat(windows, dim=-1))

    def scores(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self.encode(batch["input_ids"])
        slots = torch.arange(self.max_slots, device=encoded.device)
        fields = torch.arange(self.num_fields, device=encoded.device)
        queries = self.slot_emb(slots)[:, None, :] + self.field_emb(fields)[None, :, :]
        scores = torch.einsum("btd,sfd->bsft", encoded, queries) * self.scale
        return self._mask_scores(scores, batch)

    def _mask_scores(self, scores: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        bsz, seq_len = input_ids.shape
        query_positions = first_positions(input_ids, QUERY)
        arange = torch.arange(seq_len, device=input_ids.device)
        pre_query = arange[None, :] < query_positions[:, None]
        key_mask = (input_ids >= KEY_RANGE[0]) & (input_ids < KEY_RANGE[1])
        value_mask = (input_ids >= VALUE_RANGE[0]) & (input_ids < VALUE_RANGE[1])
        masks = [key_mask & pre_query, value_mask & pre_query]
        if self.has_condition:
            condition_mask = (input_ids >= CONDITION_RANGE[0]) & (input_ids < CONDITION_RANGE[1])
            masks.append(condition_mask & pre_query)
        field_mask = torch.stack(masks, dim=1)[:, None, :, :]
        return scores.masked_fill(~field_mask, -1.0e9)

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = self.scores(batch)
        targets = [
            batch["memory_token_positions"][:, :, 0],
            batch["memory_token_positions"][:, :, 1],
        ]
        if self.has_condition:
            targets.append(batch["memory_condition_positions"])
        valid_slots = batch["memory_mask"]
        losses = []
        for field, target in enumerate(targets):
            valid = valid_slots & (target >= 0)
            if not bool(valid.any().item()):
                continue
            field_scores = scores[:, :, field, :]
            losses.append(F.cross_entropy(field_scores[valid], target[valid]))
        if not losses:
            return scores.new_zeros(())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores = self.scores(batch)
        predicted = scores.argmax(dim=-1)
        rewritten = clone_batch(batch)
        rewritten["memory_token_positions"] = torch.stack([predicted[:, :, 0], predicted[:, :, 1]], dim=-1)
        if self.has_condition:
            rewritten["memory_condition_positions"] = predicted[:, :, 2]
        else:
            rewritten["memory_condition_positions"] = torch.full_like(predicted[:, :, 0], -1)
        starts = torch.minimum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        ends = torch.maximum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        if self.has_condition:
            starts = torch.minimum(starts, rewritten["memory_condition_positions"])
            ends = torch.maximum(ends, rewritten["memory_condition_positions"])
        rewritten["memory_spans"] = torch.stack([starts, ends], dim=-1)
        rewritten["memory_mask"] = batch["memory_mask"].clone()
        return rewritten


def _assignment_from_cost(cost: torch.Tensor) -> List[Tuple[int, int]]:
    """Return (true_index, pred_index) pairs for a small rectangular assignment.

    This is a compact Hungarian solver for the common rectangular case where
    the number of predicted slots is at least the number of true slots. Keeping
    it local avoids adding scipy while staying fast enough for per-step writer
    training.
    """

    true_count, pred_count = cost.shape
    if true_count == 0 or pred_count == 0:
        return []
    if true_count > pred_count:
        transposed = _assignment_from_cost(cost.t())
        return [(pred_idx, true_idx) for true_idx, pred_idx in transposed]

    matrix = cost.detach().cpu().double().tolist()
    n = true_count
    m = pred_count
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = []
    for pred_idx in range(1, m + 1):
        true_idx = p[pred_idx]
        if true_idx != 0:
            pairs.append((true_idx - 1, pred_idx - 1))
    return sorted(pairs)


class LearnedSetExtractorV2(nn.Module):
    """DETR-style unordered typed slot extractor with objectness."""

    def __init__(
        self,
        max_slots: int,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        objectness_threshold: float = 0.5,
        lambda_obj: float = 1.0,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.seq_len = seq_len
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.objectness_threshold = objectness_threshold
        self.lambda_obj = lambda_obj
        self.token_emb = nn.Embedding(VOCAB_SIZE, extractor_dim)
        self.pos_emb = nn.Embedding(seq_len, extractor_dim)
        self.slot_query = nn.Embedding(max_slots, extractor_dim)
        self.field_emb = nn.Embedding(3, extractor_dim)
        modules = []
        in_dim = extractor_dim * 5
        current = in_dim
        for _ in range(max(layers - 1, 0)):
            modules.append(nn.Linear(current, hidden))
            modules.append(nn.Tanh())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            current = hidden
        modules.append(nn.Linear(current, extractor_dim))
        self.context = nn.Sequential(*modules)
        self.objectness = nn.Sequential(nn.Linear(extractor_dim * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.scale = extractor_dim ** -0.5

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        windows = [_shift_with_zero(x, shift) for shift in (-2, -1, 0, 1, 2)]
        return self.context(torch.cat(windows, dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch["input_ids"])
        slots = torch.arange(self.max_slots, device=encoded.device)
        fields = torch.arange(self.num_fields, device=encoded.device)
        slot_vectors = self.slot_query(slots)
        field_queries = slot_vectors[:, None, :] + self.field_emb(fields)[None, :, :]
        field_scores = torch.einsum("btd,sfd->bsft", encoded, field_queries) * self.scale
        field_scores = self._mask_scores(field_scores, batch)
        pooled = encoded.mean(dim=1)
        object_features = torch.cat(
            [
                slot_vectors[None, :, :].expand(encoded.size(0), -1, -1),
                pooled[:, None, :].expand(-1, self.max_slots, -1),
            ],
            dim=-1,
        )
        objectness_logits = self.objectness(object_features).squeeze(-1)
        return field_scores, objectness_logits

    def _mask_scores(self, scores: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        _, seq_len = input_ids.shape
        query_positions = first_positions(input_ids, QUERY)
        arange = torch.arange(seq_len, device=input_ids.device)
        pre_query = arange[None, :] < query_positions[:, None]
        key_mask = (input_ids >= KEY_RANGE[0]) & (input_ids < KEY_RANGE[1])
        value_mask = (input_ids >= VALUE_RANGE[0]) & (input_ids < VALUE_RANGE[1])
        masks = [key_mask & pre_query, value_mask & pre_query]
        if self.has_condition:
            condition_mask = (input_ids >= CONDITION_RANGE[0]) & (input_ids < CONDITION_RANGE[1])
            masks.append(condition_mask & pre_query)
        field_mask = torch.stack(masks, dim=1)[:, None, :, :]
        return scores.masked_fill(~field_mask, -1.0e9)

    def _targets_for_sample(self, batch: Dict[str, torch.Tensor], b: int) -> List[torch.Tensor]:
        valid = batch["memory_mask"][b]
        targets = [
            batch["memory_token_positions"][b, valid, 0],
            batch["memory_token_positions"][b, valid, 1],
        ]
        if self.has_condition:
            targets.append(batch["memory_condition_positions"][b, valid])
        return targets

    def _matching_pairs_for_sample(
        self,
        batch: Dict[str, torch.Tensor],
        field_scores: torch.Tensor,
        b: int,
    ) -> List[Tuple[int, int]]:
        targets = self._targets_for_sample(batch, b)
        true_count = int(targets[0].numel())
        if true_count == 0:
            return []
        log_probs = [F.log_softmax(field_scores[b, :, field, :], dim=-1) for field in range(self.num_fields)]
        cost = torch.zeros((true_count, self.max_slots), device=field_scores.device)
        for field, target in enumerate(targets):
            cost = cost - log_probs[field][:, target].transpose(0, 1)
        return _assignment_from_cost(cost)

    def _build_rewritten(
        self,
        batch: Dict[str, torch.Tensor],
        predicted: torch.Tensor,
        active: torch.Tensor,
        probs: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        rewritten = clone_batch(batch)
        rewritten["memory_token_positions"] = torch.stack([predicted[:, :, 0], predicted[:, :, 1]], dim=-1)
        if self.has_condition:
            rewritten["memory_condition_positions"] = predicted[:, :, 2]
        else:
            rewritten["memory_condition_positions"] = torch.full_like(predicted[:, :, 0], -1)
        starts = torch.minimum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        ends = torch.maximum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        if self.has_condition:
            starts = torch.minimum(starts, rewritten["memory_condition_positions"])
            ends = torch.maximum(ends, rewritten["memory_condition_positions"])
        rewritten["memory_spans"] = torch.stack([starts, ends], dim=-1)
        rewritten["memory_mask"] = active
        rewritten["objectness_probs"] = probs
        if target_mask is not None:
            rewritten["objectness_target_mask"] = target_mask
        return rewritten

    def _gold_positions_for_slot(self, batch: Dict[str, torch.Tensor], b: int, true_idx: int) -> torch.Tensor:
        valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
        slot = int(valid[true_idx].item())
        fields = [
            batch["memory_token_positions"][b, slot, 0],
            batch["memory_token_positions"][b, slot, 1],
        ]
        if self.has_condition:
            fields.append(batch["memory_condition_positions"][b, slot])
        return torch.stack(fields)

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        field_scores, objectness_logits = self.forward(batch)
        bsz = field_scores.size(0)
        losses = []
        obj_targets = torch.zeros_like(objectness_logits)
        for b in range(bsz):
            targets = self._targets_for_sample(batch, b)
            true_count = int(targets[0].numel())
            if true_count == 0:
                continue
            log_probs = [F.log_softmax(field_scores[b, :, field, :], dim=-1) for field in range(self.num_fields)]
            cost = torch.zeros((true_count, self.max_slots), device=field_scores.device)
            for field, target in enumerate(targets):
                cost = cost - log_probs[field][:, target].transpose(0, 1)
            cost = cost + F.softplus(-objectness_logits[b])[None, :]
            pairs = _assignment_from_cost(cost)
            for true_idx, pred_idx in pairs:
                obj_targets[b, pred_idx] = 1.0
                for field, target in enumerate(targets):
                    losses.append(F.cross_entropy(field_scores[b, pred_idx, field, :][None, :], target[true_idx][None]))
        obj_loss = F.binary_cross_entropy_with_logits(objectness_logits, obj_targets)
        if not losses:
            return self.lambda_obj * obj_loss
        return torch.stack(losses).mean() + self.lambda_obj * obj_loss

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        eval_mode: str = "normal_v2",
        threshold: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        field_scores, objectness_logits = self.forward(batch)
        predicted = field_scores.argmax(dim=-1)
        probs = torch.sigmoid(objectness_logits)
        cutoff = self.objectness_threshold if threshold is None else threshold
        active = probs > cutoff
        target_mask = torch.zeros_like(active)
        pairs_by_sample = []
        for b in range(field_scores.size(0)):
            pairs = self._matching_pairs_for_sample(batch, field_scores, b)
            pairs_by_sample.append(pairs)
            for _, pred_idx in pairs:
                target_mask[b, pred_idx] = True

        if eval_mode == "normal_v2":
            return self._build_rewritten(batch, predicted, active, probs, target_mask)

        if eval_mode == "oracle_count_topk":
            active = torch.zeros_like(active)
            for b in range(probs.size(0)):
                true_count = int(batch["memory_mask"][b].sum().item())
                if true_count > 0:
                    active[b, torch.topk(probs[b], k=min(true_count, self.max_slots)).indices] = True
            return self._build_rewritten(batch, predicted, active, probs, target_mask)

        if eval_mode == "oracle_objectness":
            return self._build_rewritten(batch, predicted, target_mask.clone(), probs, target_mask)

        if eval_mode == "oracle_fields":
            oracle_predicted = predicted.clone()
            for b, pairs in enumerate(pairs_by_sample):
                for true_idx, pred_idx in pairs:
                    oracle_predicted[b, pred_idx, : self.num_fields] = self._gold_positions_for_slot(batch, b, true_idx)
            return self._build_rewritten(batch, oracle_predicted, active, probs, target_mask)

        if eval_mode == "oracle_count_and_fields":
            active = torch.zeros_like(active)
            oracle_predicted = predicted.clone()
            for b in range(probs.size(0)):
                true_count = int(batch["memory_mask"][b].sum().item())
                if true_count == 0:
                    continue
                selected = torch.topk(probs[b], k=min(true_count, self.max_slots)).indices
                active[b, selected] = True
                valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
                for out_idx, pred_idx in enumerate(selected[: len(valid)]):
                    slot = int(valid[out_idx].item())
                    oracle_predicted[b, pred_idx, 0] = batch["memory_token_positions"][b, slot, 0]
                    oracle_predicted[b, pred_idx, 1] = batch["memory_token_positions"][b, slot, 1]
                    if self.has_condition:
                        oracle_predicted[b, pred_idx, 2] = batch["memory_condition_positions"][b, slot]
            return self._build_rewritten(batch, oracle_predicted, active, probs, target_mask)

        raise ValueError(f"unknown v2 eval_mode: {eval_mode}")


def _pre_query_field_masks(input_ids: torch.Tensor, has_condition: bool) -> torch.Tensor:
    _, seq_len = input_ids.shape
    query_positions = first_positions(input_ids, QUERY)
    arange = torch.arange(seq_len, device=input_ids.device)
    pre_query = arange[None, :] < query_positions[:, None]
    key_mask = (input_ids >= KEY_RANGE[0]) & (input_ids < KEY_RANGE[1]) & pre_query
    value_mask = (input_ids >= VALUE_RANGE[0]) & (input_ids < VALUE_RANGE[1]) & pre_query
    masks = [key_mask, value_mask]
    if has_condition:
        cond_mask = (input_ids >= CONDITION_RANGE[0]) & (input_ids < CONDITION_RANGE[1]) & pre_query
        masks.append(cond_mask)
    return torch.stack(masks, dim=1)


def _gold_field_position_lists(
    batch: Dict[str, torch.Tensor],
    has_condition: bool,
    b: int,
) -> List[List[int]]:
    valid = batch["memory_mask"][b]
    fields = [
        batch["memory_token_positions"][b, valid, 0],
        batch["memory_token_positions"][b, valid, 1],
    ]
    if has_condition:
        fields.append(batch["memory_condition_positions"][b, valid])
    return [[int(pos.item()) for pos in field if int(pos.item()) >= 0] for field in fields]


CandidateK = int | Sequence[int]


def _candidate_k_by_field(candidate_k: CandidateK, num_fields: int) -> List[int]:
    if isinstance(candidate_k, int):
        return [max(1, int(candidate_k)) for _ in range(num_fields)]
    values = [max(1, int(value)) for value in candidate_k]
    if len(values) != num_fields:
        raise ValueError(f"candidate_k must provide {num_fields} field budgets, got {values}")
    return values


class CandidateFieldProposer(nn.Module):
    """High-recall field candidate scorer for pre-query typed tokens."""

    def __init__(
        self,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.token_emb = nn.Embedding(VOCAB_SIZE, extractor_dim)
        self.pos_emb = nn.Embedding(seq_len, extractor_dim)
        modules = []
        current = extractor_dim * 5
        for _ in range(max(layers - 1, 0)):
            modules.append(nn.Linear(current, hidden))
            modules.append(nn.Tanh())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            current = hidden
        modules.append(nn.Linear(current, extractor_dim))
        self.context = nn.Sequential(*modules)
        self.field_heads = nn.Linear(extractor_dim, self.num_fields)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        windows = [_shift_with_zero(x, shift) for shift in (-2, -1, 0, 1, 2)]
        return self.context(torch.cat(windows, dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch["input_ids"])
        logits = self.field_heads(encoded).transpose(1, 2)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        return logits.masked_fill(~masks, -1.0e9), encoded

    def target_masks(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len = batch["input_ids"].shape
        targets = torch.zeros((bsz, self.num_fields, seq_len), device=batch["input_ids"].device)
        for b in range(bsz):
            fields = _gold_field_position_lists(batch, self.has_condition, b)
            for field, positions in enumerate(fields):
                if positions:
                    targets[b, field, torch.tensor(positions, device=targets.device)] = 1.0
        return targets

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, _ = self.forward(batch)
        targets = self.target_masks(batch)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        losses = []
        for field in range(self.num_fields):
            valid = masks[:, field]
            if bool(valid.any().item()):
                losses.append(F.binary_cross_entropy_with_logits(logits[:, field, :][valid], targets[:, field, :][valid]))
        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def candidate_pools(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, encoded = self.forward(batch)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        bsz = batch["input_ids"].size(0)
        field_ks = _candidate_k_by_field(candidate_k, self.num_fields)
        max_candidate_k = max(field_ks)
        positions = torch.full((bsz, self.num_fields, max_candidate_k), -1, device=batch["input_ids"].device, dtype=torch.long)
        candidate_mask = torch.zeros((bsz, self.num_fields, max_candidate_k), device=batch["input_ids"].device, dtype=torch.bool)
        for b in range(bsz):
            gold_fields = _gold_field_position_lists(batch, self.has_condition, b)
            for field in range(self.num_fields):
                field_k = field_ks[field]
                valid_positions = torch.nonzero(masks[b, field], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                field_logits = logits[b, field].clone()
                selected: List[int] = []
                if mode in {"oracle_candidates", "oracle_candidates_plus_noise", "oracle_candidates_gold_only"}:
                    for pos in gold_fields[field]:
                        if pos not in selected and len(selected) < field_k:
                            selected.append(pos)
                if mode == "learned_candidates":
                    pool = torch.topk(field_logits, k=min(field_k, int(valid_positions.numel()))).indices.tolist()
                    selected.extend(int(pos) for pos in pool if int(pos) not in selected)
                elif mode != "oracle_candidates_gold_only":
                    for pos in gold_fields[field]:
                        field_logits[pos] = -1.0e9
                    fill_count = field_k - len(selected)
                    if fill_count > 0:
                        top = torch.topk(field_logits, k=min(fill_count, int(valid_positions.numel()))).indices.tolist()
                        selected.extend(int(pos) for pos in top if int(pos) not in selected)
                if selected:
                    selected = selected[:field_k]
                    positions[b, field, : len(selected)] = torch.tensor(selected, device=positions.device, dtype=torch.long)
                    candidate_mask[b, field, : len(selected)] = True
        return positions, candidate_mask, logits, encoded


class ConditionCandidateProposerV2(CandidateFieldProposer):
    """Condition-focused candidate proposer for heldout template generalization.

    The tuple scorer still consumes token positions. The v2 proposer improves
    the condition-token ranking with span evidence, lightweight condition-role
    contrast, and a fact-local auxiliary loss. These are diagnostic additions
    around the writer, not a larger backbone.
    """

    def __init__(
        self,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        span_condition: bool = False,
        guideline_condition: bool = False,
        simplified_aux_weight: float = 0.0,
        guideline_loss_weight: float = 0.0,
        max_span_width: int = 4,
    ):
        super().__init__(seq_len, has_condition, extractor_dim, hidden, layers, dropout)
        self.span_condition = bool(span_condition and has_condition)
        self.guideline_condition = bool(guideline_condition and has_condition)
        self.simplified_aux_weight = float(simplified_aux_weight if has_condition else 0.0)
        self.guideline_loss_weight = float(guideline_loss_weight if has_condition else 0.0)
        self.max_span_width = max(1, int(max_span_width))
        if self.span_condition:
            self.width_emb = nn.Embedding(self.max_span_width + 1, extractor_dim)
            self.span_scorer = nn.Sequential(
                nn.Linear(extractor_dim * 7, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.span_scale = nn.Parameter(torch.tensor(1.0))
        if self.guideline_condition:
            self.role_emb = nn.Embedding(CONDITION_RANGE[1] - CONDITION_RANGE[0], extractor_dim)
            self.role_scorer = nn.Sequential(
                nn.Linear(extractor_dim * 4, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            self.guideline_scale = nn.Parameter(torch.tensor(0.5))
        self.last_candidate_debug: Dict[str, float] = {}

    def _span_token_logits(
        self,
        encoded: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = encoded.shape
        aligned_scores: List[torch.Tensor] = []
        positive_span_scores = encoded.new_zeros(())
        prefix = torch.cat([encoded.new_zeros(bsz, 1, dim), encoded.cumsum(dim=1)], dim=1)
        query_positions = first_positions(input_ids, QUERY)
        arange = torch.arange(seq_len, device=encoded.device)
        pre_query = arange[None, :] < query_positions[:, None]
        for width in range(1, self.max_span_width + 1):
            count = seq_len - width + 1
            if count <= 0:
                continue
            start = torch.arange(count, device=encoded.device)
            end = start + width - 1
            left = encoded[:, start]
            right = encoded[:, end]
            mean = (prefix[:, width:] - prefix[:, :-width]) / float(width)
            pieces = torch.stack([encoded[:, offset : offset + count] for offset in range(width)], dim=2)
            max_pool = pieces.max(dim=2).values
            left_ctx = encoded[:, (start - 1).clamp_min(0)]
            right_ctx = encoded[:, (end + 1).clamp_max(seq_len - 1)]
            width_vec = self.width_emb(torch.full((count,), width, device=encoded.device, dtype=torch.long))
            width_vec = width_vec[None, :, :].expand(bsz, -1, -1)
            features = torch.cat([left, right, mean, max_pool, width_vec, left_ctx, right_ctx], dim=-1)
            scores = self.span_scorer(features).squeeze(-1)
            span_valid = pre_query[:, start] & pre_query[:, end]
            scores = scores.masked_fill(~span_valid, -1.0e9)
            for offset in range(width):
                aligned = encoded.new_full((bsz, seq_len), -1.0e9)
                aligned[:, offset : offset + count] = scores
                aligned_scores.append(aligned)
        token_scores = torch.stack(aligned_scores, dim=0).amax(dim=0) if aligned_scores else encoded.new_full((bsz, seq_len), -1.0e9)
        return token_scores, positive_span_scores

    def _guideline_logits(self, encoded: torch.Tensor) -> torch.Tensor:
        role = self.role_emb.weight
        token = encoded[:, :, None, :].expand(-1, -1, role.size(0), -1)
        role_expanded = role[None, None, :, :].expand(encoded.size(0), encoded.size(1), -1, -1)
        features = torch.cat(
            [
                token,
                role_expanded,
                token * role_expanded,
                (token - role_expanded).abs(),
            ],
            dim=-1,
        )
        return self.role_scorer(features).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch["input_ids"])
        logits = self.field_heads(encoded).transpose(1, 2)
        if self.has_condition and self.span_condition:
            span_token_logits, _ = self._span_token_logits(encoded, batch["input_ids"])
            logits[:, 2, :] = logits[:, 2, :] + self.span_scale * span_token_logits.clamp_min(-20.0)
        if self.has_condition and self.guideline_condition:
            role_logits = self._guideline_logits(encoded)
            condition_boost = torch.logsumexp(role_logits, dim=-1) - torch.log(
                torch.tensor(float(role_logits.size(-1)), device=encoded.device)
            )
            logits[:, 2, :] = logits[:, 2, :] + self.guideline_scale * condition_boost
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        return logits.masked_fill(~masks, -1.0e9), encoded

    def _span_loss(self, batch: Dict[str, torch.Tensor], encoded: torch.Tensor) -> torch.Tensor:
        if not self.span_condition:
            return encoded.new_zeros(())
        span_logits, _ = self._span_token_logits(encoded, batch["input_ids"])
        targets = self.target_masks(batch)[:, 2, :]
        valid = _pre_query_field_masks(batch["input_ids"], self.has_condition)[:, 2, :]
        if not bool(valid.any().item()):
            return encoded.new_zeros(())
        return F.binary_cross_entropy_with_logits(span_logits[valid], targets[valid])

    def _guideline_loss(self, batch: Dict[str, torch.Tensor], encoded: torch.Tensor) -> torch.Tensor:
        if not self.guideline_condition:
            return encoded.new_zeros(())
        role_logits = self._guideline_logits(encoded)
        losses = []
        for b in range(batch["input_ids"].size(0)):
            valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            for slot_tensor in valid:
                pos = int(batch["memory_condition_positions"][b, int(slot_tensor.item())].item())
                if pos < 0:
                    continue
                token = int(batch["input_ids"][b, pos].item())
                if CONDITION_RANGE[0] <= token < CONDITION_RANGE[1]:
                    target = torch.tensor([token - CONDITION_RANGE[0]], device=encoded.device)
                    losses.append(F.cross_entropy(role_logits[b, pos][None, :], target))
        return encoded.new_zeros(()) if not losses else torch.stack(losses).mean()

    def _simplified_aux_loss(self, batch: Dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        if self.simplified_aux_weight <= 0.0 or not self.has_condition:
            return logits.new_zeros(())
        losses = []
        for b in range(batch["input_ids"].size(0)):
            valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            for slot_tensor in valid:
                slot = int(slot_tensor.item())
                cond_pos = int(batch["memory_condition_positions"][b, slot].item())
                if cond_pos < 0:
                    continue
                start = int(batch["memory_spans"][b, slot, 0].item())
                end = int(batch["memory_spans"][b, slot, 1].item())
                start = max(0, min(start, logits.size(-1) - 1))
                end = max(start, min(end, logits.size(-1) - 1))
                local_logits = logits[b, 2, start : end + 1]
                local_targets = torch.zeros_like(local_logits)
                if start <= cond_pos <= end:
                    local_targets[cond_pos - start] = 1.0
                losses.append(F.binary_cross_entropy_with_logits(local_logits, local_targets))
        return logits.new_zeros(()) if not losses else torch.stack(losses).mean()

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, encoded = self.forward(batch)
        targets = self.target_masks(batch)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        losses = []
        for field in range(self.num_fields):
            valid = masks[:, field]
            if bool(valid.any().item()):
                losses.append(F.binary_cross_entropy_with_logits(logits[:, field, :][valid], targets[:, field, :][valid]))
        base = logits.new_zeros(()) if not losses else torch.stack(losses).mean()
        span = self._span_loss(batch, encoded)
        guideline = self._guideline_loss(batch, encoded)
        simplified = self._simplified_aux_loss(batch, logits)
        return base + span + self.guideline_loss_weight * guideline + self.simplified_aux_weight * simplified

    @torch.no_grad()
    def candidate_pools(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions, candidate_mask, logits, encoded = super().candidate_pools(batch, candidate_k, mode)
        self.last_candidate_debug = {}
        if not self.has_condition:
            return positions, candidate_mask, logits, encoded

        field_ks = _candidate_k_by_field(candidate_k, self.num_fields)
        condition_k = field_ks[2]
        base_logits = self.field_heads(encoded).transpose(1, 2)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        token_hits = 0
        span_hits = 0
        any_hits = 0
        gold_total = 0
        span_total = 0
        span_false = 0
        span_token_logits = None
        if self.span_condition:
            span_token_logits, _ = self._span_token_logits(encoded, batch["input_ids"])
        for b in range(batch["input_ids"].size(0)):
            gold_conditions = _gold_field_position_lists(batch, self.has_condition, b)[2]
            gold_set = set(gold_conditions)
            gold_total += len(gold_conditions)
            valid_positions = torch.nonzero(masks[b, 2], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            token_top = torch.topk(base_logits[b, 2].masked_fill(~masks[b, 2], -1.0e9), k=min(condition_k, int(valid_positions.numel()))).indices.tolist()
            token_set = {int(pos) for pos in token_top}
            final_set = {
                int(pos.item())
                for idx, pos in enumerate(positions[b, 2])
                if bool(candidate_mask[b, 2, idx].item())
            }
            if span_token_logits is not None:
                span_top = torch.topk(span_token_logits[b].masked_fill(~masks[b, 2], -1.0e9), k=min(condition_k, int(valid_positions.numel()))).indices.tolist()
                span_set = {int(pos) for pos in span_top}
            else:
                span_set = set()
            token_hits += len([pos for pos in gold_conditions if pos in token_set])
            span_hits += len([pos for pos in gold_conditions if pos in span_set])
            any_hits += len([pos for pos in gold_conditions if pos in final_set])
            span_total += len(span_set)
            span_false += len([pos for pos in span_set if pos not in gold_set])
        self.last_candidate_debug = {
            "condition_token_recall": token_hits / max(gold_total, 1),
            "condition_span_recall": span_hits / max(gold_total, 1),
            "condition_any_recall": any_hits / max(gold_total, 1),
            "condition_span_precision": (span_total - span_false) / max(span_total, 1),
        }
        return positions, candidate_mask, logits, encoded


class ConditionCandidateProposerV3(ConditionCandidateProposerV2):
    """High-recall condition candidate union for the stress setting.

    V3 keeps the v4.5 span/guideline/simplified losses, then forms condition
    pools by interleaving several ranked views: token logits, span logits,
    guideline-role logits, and a separate local-window condition scorer. Only
    pre-query typed condition tokens are eligible.
    """

    def __init__(
        self,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        span_condition: bool = True,
        guideline_condition: bool = True,
        simplified_aux_weight: float = 0.5,
        guideline_loss_weight: float = 2.0,
        max_span_width: int = 4,
    ):
        super().__init__(
            seq_len,
            has_condition,
            extractor_dim,
            hidden,
            layers,
            dropout,
            span_condition=span_condition,
            guideline_condition=guideline_condition,
            simplified_aux_weight=simplified_aux_weight,
            guideline_loss_weight=guideline_loss_weight,
            max_span_width=max_span_width,
        )
        if not has_condition:
            raise ValueError("ConditionCandidateProposerV3 is only for conditional slots")
        self.local_condition_scorer = nn.Sequential(
            nn.Linear(extractor_dim * 5, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.local_condition_scale = nn.Parameter(torch.tensor(0.5))

    def _local_condition_logits(self, encoded: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        features = torch.cat([_shift_with_zero(encoded, shift) for shift in (-2, -1, 0, 1, 2)], dim=-1)
        logits = self.local_condition_scorer(features).squeeze(-1)
        masks = _pre_query_field_masks(input_ids, self.has_condition)[:, 2, :]
        return logits.masked_fill(~masks, -1.0e9)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, encoded = super().forward(batch)
        local_logits = self._local_condition_logits(encoded, batch["input_ids"])
        logits = logits.clone()
        logits[:, 2, :] = logits[:, 2, :] + self.local_condition_scale * local_logits.clamp_min(-20.0)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        return logits.masked_fill(~masks, -1.0e9), encoded

    @torch.no_grad()
    def candidate_pools(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions, candidate_mask, logits, encoded = super().candidate_pools(batch, candidate_k, mode)
        if mode != "learned_candidates":
            return positions, candidate_mask, logits, encoded

        field_ks = _candidate_k_by_field(candidate_k, self.num_fields)
        condition_k = field_ks[2]
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        base_logits = self.field_heads(encoded).transpose(1, 2)
        span_logits = self._span_token_logits(encoded, batch["input_ids"])[0] if self.span_condition else None
        if self.guideline_condition:
            role_logits = self._guideline_logits(encoded)
            role_boost = torch.logsumexp(role_logits, dim=-1) - torch.log(
                torch.tensor(float(role_logits.size(-1)), device=encoded.device)
            )
        else:
            role_boost = None
        local_logits = self._local_condition_logits(encoded, batch["input_ids"])

        token_hits = span_hits = role_hits = local_hits = union_hits = 0
        gold_total = 0
        total_selected = 0
        false_selected = 0
        for b in range(batch["input_ids"].size(0)):
            valid_positions = torch.nonzero(masks[b, 2], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            gold_conditions = _gold_field_position_lists(batch, self.has_condition, b)[2]
            gold_set = set(gold_conditions)
            gold_total += len(gold_conditions)
            sources = [
                logits[b, 2],
                base_logits[b, 2],
            ]
            if span_logits is not None:
                sources.append(span_logits[b])
            if role_boost is not None:
                sources.append(role_boost[b])
            sources.append(local_logits[b])

            ranked_lists: List[List[int]] = []
            for source in sources:
                masked = source.masked_fill(~masks[b, 2], -1.0e9)
                ranked_lists.append(
                    [int(pos) for pos in torch.topk(masked, k=min(condition_k, int(valid_positions.numel()))).indices.tolist()]
                )

            selected: List[int] = []
            for rank in range(condition_k):
                for ranked in ranked_lists:
                    if rank < len(ranked) and ranked[rank] not in selected:
                        selected.append(ranked[rank])
                    if len(selected) >= condition_k:
                        break
                if len(selected) >= condition_k:
                    break
            if len(selected) < condition_k:
                for pos in ranked_lists[0]:
                    if pos not in selected:
                        selected.append(pos)
                    if len(selected) >= condition_k:
                        break

            positions[b, 2, :] = -1
            candidate_mask[b, 2, :] = False
            if selected:
                selected = selected[:condition_k]
                positions[b, 2, : len(selected)] = torch.tensor(selected, device=positions.device, dtype=torch.long)
                candidate_mask[b, 2, : len(selected)] = True

            token_set = set(ranked_lists[1] if len(ranked_lists) > 1 else [])
            span_set = set(ranked_lists[2] if span_logits is not None and len(ranked_lists) > 2 else [])
            role_index = 3 if span_logits is not None else 2
            role_set = set(ranked_lists[role_index]) if role_boost is not None and len(ranked_lists) > role_index else set()
            local_set = set(ranked_lists[-1])
            union_set = set(selected)
            token_hits += len([pos for pos in gold_conditions if pos in token_set])
            span_hits += len([pos for pos in gold_conditions if pos in span_set])
            role_hits += len([pos for pos in gold_conditions if pos in role_set])
            local_hits += len([pos for pos in gold_conditions if pos in local_set])
            union_hits += len([pos for pos in gold_conditions if pos in union_set])
            total_selected += len(union_set)
            false_selected += len([pos for pos in union_set if pos not in gold_set])

        self.last_candidate_debug = {
            **self.last_candidate_debug,
            "condition_v3_token_recall": token_hits / max(gold_total, 1),
            "condition_v3_span_recall": span_hits / max(gold_total, 1),
            "condition_v3_role_recall": role_hits / max(gold_total, 1),
            "condition_v3_local_recall": local_hits / max(gold_total, 1),
            "condition_v3_union_recall": union_hits / max(gold_total, 1),
            "condition_v3_union_precision": (total_selected - false_selected) / max(total_selected, 1),
            "condition_pool_size": total_selected / max(batch["input_ids"].size(0), 1),
        }
        return positions, candidate_mask, logits, encoded


class CoexistingCandidateProposerV2(CandidateFieldProposer):
    """Key/value candidate proposer for heldout coexisting templates."""

    def __init__(
        self,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        span_fields: bool = True,
        guideline_fields: bool = True,
        simplified_aux_weight: float = 0.5,
        guideline_loss_weight: float = 2.0,
        max_span_width: int = 4,
    ):
        super().__init__(seq_len, has_condition, extractor_dim, hidden, layers, dropout)
        if has_condition:
            raise ValueError("CoexistingCandidateProposerV2 is for key/value tasks only")
        self.span_fields = bool(span_fields)
        self.guideline_fields = bool(guideline_fields)
        self.simplified_aux_weight = float(simplified_aux_weight)
        self.guideline_loss_weight = float(guideline_loss_weight)
        self.max_span_width = max(1, int(max_span_width))
        if self.span_fields:
            self.width_emb = nn.Embedding(self.max_span_width + 1, extractor_dim)
            self.span_scorers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(extractor_dim * 7, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                    for _ in range(2)
                ]
            )
            self.span_scale = nn.Parameter(torch.ones(2))
        if self.guideline_fields:
            self.key_role_emb = nn.Embedding(KEY_RANGE[1] - KEY_RANGE[0], extractor_dim)
            self.value_role_emb = nn.Embedding(VALUE_RANGE[1] - VALUE_RANGE[0], extractor_dim)
            self.role_scorers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(extractor_dim * 4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                    for _ in range(2)
                ]
            )
            self.guideline_scale = nn.Parameter(torch.full((2,), 0.5))
        self.last_candidate_debug: Dict[str, float] = {}

    def _span_token_logits(
        self,
        encoded: torch.Tensor,
        input_ids: torch.Tensor,
        field: int,
    ) -> torch.Tensor:
        bsz, seq_len, dim = encoded.shape
        aligned_scores: List[torch.Tensor] = []
        prefix = torch.cat([encoded.new_zeros(bsz, 1, dim), encoded.cumsum(dim=1)], dim=1)
        query_positions = first_positions(input_ids, QUERY)
        arange = torch.arange(seq_len, device=encoded.device)
        pre_query = arange[None, :] < query_positions[:, None]
        scorer = self.span_scorers[field]
        for width in range(1, self.max_span_width + 1):
            count = seq_len - width + 1
            if count <= 0:
                continue
            start = torch.arange(count, device=encoded.device)
            end = start + width - 1
            left = encoded[:, start]
            right = encoded[:, end]
            mean = (prefix[:, width:] - prefix[:, :-width]) / float(width)
            pieces = torch.stack([encoded[:, offset : offset + count] for offset in range(width)], dim=2)
            max_pool = pieces.max(dim=2).values
            left_ctx = encoded[:, (start - 1).clamp_min(0)]
            right_ctx = encoded[:, (end + 1).clamp_max(seq_len - 1)]
            width_vec = self.width_emb(torch.full((count,), width, device=encoded.device, dtype=torch.long))
            width_vec = width_vec[None, :, :].expand(bsz, -1, -1)
            features = torch.cat([left, right, mean, max_pool, width_vec, left_ctx, right_ctx], dim=-1)
            scores = scorer(features).squeeze(-1)
            scores = scores.masked_fill(~(pre_query[:, start] & pre_query[:, end]), -1.0e9)
            for offset in range(width):
                aligned = encoded.new_full((bsz, seq_len), -1.0e9)
                aligned[:, offset : offset + count] = scores
                aligned_scores.append(aligned)
        return torch.stack(aligned_scores, dim=0).amax(dim=0) if aligned_scores else encoded.new_full((bsz, seq_len), -1.0e9)

    def _role_logits(self, encoded: torch.Tensor, field: int) -> torch.Tensor:
        role = self.key_role_emb.weight if field == 0 else self.value_role_emb.weight
        scorer = self.role_scorers[field]
        token = encoded[:, :, None, :].expand(-1, -1, role.size(0), -1)
        role_expanded = role[None, None, :, :].expand(encoded.size(0), encoded.size(1), -1, -1)
        features = torch.cat([token, role_expanded, token * role_expanded, (token - role_expanded).abs()], dim=-1)
        return scorer(features).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(batch["input_ids"])
        logits = self.field_heads(encoded).transpose(1, 2)
        if self.span_fields:
            for field in range(2):
                span_logits = self._span_token_logits(encoded, batch["input_ids"], field)
                logits[:, field, :] = logits[:, field, :] + self.span_scale[field] * span_logits.clamp_min(-20.0)
        if self.guideline_fields:
            for field in range(2):
                role_logits = self._role_logits(encoded, field)
                role_boost = torch.logsumexp(role_logits, dim=-1) - torch.log(
                    torch.tensor(float(role_logits.size(-1)), device=encoded.device)
                )
                logits[:, field, :] = logits[:, field, :] + self.guideline_scale[field] * role_boost
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        return logits.masked_fill(~masks, -1.0e9), encoded

    def _span_loss(self, batch: Dict[str, torch.Tensor], encoded: torch.Tensor) -> torch.Tensor:
        if not self.span_fields:
            return encoded.new_zeros(())
        targets = self.target_masks(batch)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        losses = []
        for field in range(2):
            span_logits = self._span_token_logits(encoded, batch["input_ids"], field)
            valid = masks[:, field]
            if bool(valid.any().item()):
                losses.append(F.binary_cross_entropy_with_logits(span_logits[valid], targets[:, field, :][valid]))
        return encoded.new_zeros(()) if not losses else torch.stack(losses).mean()

    def _guideline_loss(self, batch: Dict[str, torch.Tensor], encoded: torch.Tensor) -> torch.Tensor:
        if not self.guideline_fields:
            return encoded.new_zeros(())
        losses = []
        ranges = [KEY_RANGE, VALUE_RANGE]
        for field in range(2):
            role_logits = self._role_logits(encoded, field)
            start, end = ranges[field]
            positions_by_field = batch["memory_token_positions"][:, :, field]
            for b in range(batch["input_ids"].size(0)):
                valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
                for slot_tensor in valid:
                    pos = int(positions_by_field[b, int(slot_tensor.item())].item())
                    if pos < 0:
                        continue
                    token = int(batch["input_ids"][b, pos].item())
                    if start <= token < end:
                        target = torch.tensor([token - start], device=encoded.device)
                        losses.append(F.cross_entropy(role_logits[b, pos][None, :], target))
        return encoded.new_zeros(()) if not losses else torch.stack(losses).mean()

    def _simplified_aux_loss(self, batch: Dict[str, torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
        if self.simplified_aux_weight <= 0.0:
            return logits.new_zeros(())
        losses = []
        for b in range(batch["input_ids"].size(0)):
            valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            for slot_tensor in valid:
                slot = int(slot_tensor.item())
                start = int(batch["memory_spans"][b, slot, 0].item())
                end = int(batch["memory_spans"][b, slot, 1].item())
                start = max(0, min(start, logits.size(-1) - 1))
                end = max(start, min(end, logits.size(-1) - 1))
                for field in range(2):
                    pos = int(batch["memory_token_positions"][b, slot, field].item())
                    local_logits = logits[b, field, start : end + 1]
                    local_targets = torch.zeros_like(local_logits)
                    if start <= pos <= end:
                        local_targets[pos - start] = 1.0
                    losses.append(F.binary_cross_entropy_with_logits(local_logits, local_targets))
        return logits.new_zeros(()) if not losses else torch.stack(losses).mean()

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits, encoded = self.forward(batch)
        targets = self.target_masks(batch)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        losses = []
        for field in range(2):
            valid = masks[:, field]
            if bool(valid.any().item()):
                losses.append(F.binary_cross_entropy_with_logits(logits[:, field, :][valid], targets[:, field, :][valid]))
        base = logits.new_zeros(()) if not losses else torch.stack(losses).mean()
        span = self._span_loss(batch, encoded)
        guideline = self._guideline_loss(batch, encoded)
        simplified = self._simplified_aux_loss(batch, logits)
        return base + span + self.guideline_loss_weight * guideline + self.simplified_aux_weight * simplified

    @torch.no_grad()
    def candidate_pools(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions, candidate_mask, logits, encoded = super().candidate_pools(batch, candidate_k, mode)
        self.last_candidate_debug = {}
        field_ks = _candidate_k_by_field(candidate_k, self.num_fields)
        base_logits = self.field_heads(encoded).transpose(1, 2)
        masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
        token_hits = [0, 0]
        span_hits = [0, 0]
        any_hits = [0, 0]
        gold_total = [0, 0]
        span_total = [0, 0]
        span_false = [0, 0]
        span_logits_by_field = [
            self._span_token_logits(encoded, batch["input_ids"], field) if self.span_fields else None
            for field in range(2)
        ]
        for b in range(batch["input_ids"].size(0)):
            gold_fields = _gold_field_position_lists(batch, self.has_condition, b)
            for field in range(2):
                field_k = field_ks[field]
                gold = gold_fields[field]
                gold_set = set(gold)
                gold_total[field] += len(gold)
                valid_positions = torch.nonzero(masks[b, field], as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                token_top = torch.topk(base_logits[b, field].masked_fill(~masks[b, field], -1.0e9), k=min(field_k, int(valid_positions.numel()))).indices.tolist()
                token_set = {int(pos) for pos in token_top}
                final_set = {
                    int(pos.item())
                    for idx, pos in enumerate(positions[b, field])
                    if bool(candidate_mask[b, field, idx].item())
                }
                if span_logits_by_field[field] is not None:
                    span_top = torch.topk(span_logits_by_field[field][b].masked_fill(~masks[b, field], -1.0e9), k=min(field_k, int(valid_positions.numel()))).indices.tolist()
                    span_set = {int(pos) for pos in span_top}
                else:
                    span_set = set()
                token_hits[field] += len([pos for pos in gold if pos in token_set])
                span_hits[field] += len([pos for pos in gold if pos in span_set])
                any_hits[field] += len([pos for pos in gold if pos in final_set])
                span_total[field] += len(span_set)
                span_false[field] += len([pos for pos in span_set if pos not in gold_set])
        self.last_candidate_debug = {
            "key_token_recall": token_hits[0] / max(gold_total[0], 1),
            "value_token_recall": token_hits[1] / max(gold_total[1], 1),
            "key_span_recall": span_hits[0] / max(gold_total[0], 1),
            "value_span_recall": span_hits[1] / max(gold_total[1], 1),
            "key_any_recall": any_hits[0] / max(gold_total[0], 1),
            "value_any_recall": any_hits[1] / max(gold_total[1], 1),
            "key_span_precision": (span_total[0] - span_false[0]) / max(span_total[0], 1),
            "value_span_precision": (span_total[1] - span_false[1]) / max(span_total[1], 1),
        }
        return positions, candidate_mask, logits, encoded


class WriterV3CandidateAssembler(nn.Module):
    """Slot set predictor over high-recall field candidate pools."""

    def __init__(
        self,
        max_slots: int,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        objectness_threshold: float = 0.5,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.seq_len = seq_len
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.objectness_threshold = objectness_threshold
        self.proposer = CandidateFieldProposer(seq_len, has_condition, extractor_dim, hidden, layers, dropout)
        self.slot_query = nn.Embedding(max_slots, extractor_dim)
        self.field_emb = nn.Embedding(3, extractor_dim)
        self.objectness = nn.Sequential(nn.Linear(extractor_dim * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.scale = extractor_dim ** -0.5

    def _candidate_states(self, encoded: torch.Tensor, candidate_positions: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = encoded.shape
        safe = candidate_positions.clamp_min(0).clamp_max(seq_len - 1)
        expanded_encoded = encoded[:, None, :, :].expand(-1, self.num_fields, -1, -1)
        gather_index = safe[..., None].expand(-1, -1, -1, dim)
        states = torch.gather(expanded_encoded, dim=2, index=gather_index)
        return torch.where(candidate_positions[..., None] >= 0, states, torch.zeros_like(states))

    def forward_with_candidates(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: int,
        candidate_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_positions, candidate_mask, _, encoded = self.proposer.candidate_pools(batch, candidate_k, candidate_mode)
        candidate_states = self._candidate_states(encoded, candidate_positions)
        slots = torch.arange(self.max_slots, device=encoded.device)
        fields = torch.arange(self.num_fields, device=encoded.device)
        field_queries = self.slot_query(slots)[:, None, :] + self.field_emb(fields)[None, :, :]
        field_scores = torch.einsum("bfkd,sfd->bsfk", candidate_states, field_queries) * self.scale
        field_scores = field_scores.masked_fill(~candidate_mask[:, None, :, :], -1.0e9)
        pooled = encoded.mean(dim=1)
        slot_vectors = self.slot_query(slots)
        object_features = torch.cat(
            [
                slot_vectors[None, :, :].expand(encoded.size(0), -1, -1),
                pooled[:, None, :].expand(-1, self.max_slots, -1),
            ],
            dim=-1,
        )
        objectness_logits = self.objectness(object_features).squeeze(-1)
        return field_scores, objectness_logits, candidate_positions, candidate_mask, encoded

    def _target_candidate_indices(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_positions: torch.Tensor,
        b: int,
    ) -> Tuple[List[List[int]], List[int]]:
        valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
        target_indices: List[List[int]] = []
        source_slots: List[int] = []
        for slot_tensor in valid:
            slot = int(slot_tensor.item())
            targets = [
                int(batch["memory_token_positions"][b, slot, 0].item()),
                int(batch["memory_token_positions"][b, slot, 1].item()),
            ]
            if self.has_condition:
                targets.append(int(batch["memory_condition_positions"][b, slot].item()))
            indices = []
            missing = False
            for field, target in enumerate(targets):
                matches = torch.nonzero(candidate_positions[b, field] == target, as_tuple=False).flatten()
                if matches.numel() == 0:
                    missing = True
                    break
                indices.append(int(matches[0].item()))
            if not missing:
                target_indices.append(indices)
                source_slots.append(slot)
        return target_indices, source_slots

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: int,
        candidate_loss_weight: float = 1.0,
        tuple_loss_weight: float = 1.0,
        rank_loss_weight: float = 0.5,
    ) -> torch.Tensor:
        candidate_loss = self.proposer.loss(batch)
        field_scores, objectness_logits, candidate_positions, _, _ = self.forward_with_candidates(
            batch,
            candidate_k,
            "oracle_candidates_plus_noise",
        )
        losses = []
        obj_targets = torch.zeros_like(objectness_logits)
        for b in range(field_scores.size(0)):
            target_indices, _ = self._target_candidate_indices(batch, candidate_positions, b)
            true_count = len(target_indices)
            if true_count == 0:
                continue
            log_probs = [F.log_softmax(field_scores[b, :, field, :], dim=-1) for field in range(self.num_fields)]
            cost = torch.zeros((true_count, self.max_slots), device=field_scores.device)
            for true_idx, indices in enumerate(target_indices):
                for field, candidate_idx in enumerate(indices):
                    cost[true_idx] = cost[true_idx] - log_probs[field][:, candidate_idx]
            cost = cost + F.softplus(-objectness_logits[b])[None, :]
            pairs = _assignment_from_cost(cost)
            for true_idx, pred_idx in pairs:
                obj_targets[b, pred_idx] = 1.0
                for field, candidate_idx in enumerate(target_indices[true_idx]):
                    losses.append(F.cross_entropy(field_scores[b, pred_idx, field, :][None, :], torch.tensor([candidate_idx], device=field_scores.device)))
        obj_loss = F.binary_cross_entropy_with_logits(objectness_logits, obj_targets)
        assembly_loss = obj_loss if not losses else torch.stack(losses).mean() + obj_loss
        return candidate_loss_weight * candidate_loss + assembly_loss

    def _candidate_cost_matrix(
        self,
        field_scores: torch.Tensor,
        objectness_logits: torch.Tensor,
        target_indices: List[List[int]],
        b: int,
        include_objectness: bool = True,
    ) -> torch.Tensor:
        true_count = len(target_indices)
        cost = torch.zeros((true_count, self.max_slots), device=field_scores.device)
        if true_count == 0:
            return cost
        log_probs = [F.log_softmax(field_scores[b, :, field, :], dim=-1) for field in range(self.num_fields)]
        for true_idx, indices in enumerate(target_indices):
            for field, candidate_idx in enumerate(indices):
                cost[true_idx] = cost[true_idx] - log_probs[field][:, candidate_idx]
        if include_objectness:
            cost = cost + F.softplus(-objectness_logits[b])[None, :]
        return cost

    def _with_gold_fields(
        self,
        predicted_positions: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        fields_to_replace: List[int],
    ) -> torch.Tensor:
        out = predicted_positions.clone()
        for b in range(batch["input_ids"].size(0)):
            valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            for out_idx, slot_tensor in enumerate(valid[: self.max_slots]):
                slot = int(slot_tensor.item())
                if 0 in fields_to_replace:
                    out[b, out_idx, 0] = batch["memory_token_positions"][b, slot, 0]
                if 1 in fields_to_replace:
                    out[b, out_idx, 1] = batch["memory_token_positions"][b, slot, 1]
                if self.has_condition and 2 in fields_to_replace:
                    out[b, out_idx, 2] = batch["memory_condition_positions"][b, slot]
        return out

    def _debug_metric_dict(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_positions: torch.Tensor,
        predicted_candidate_indices: torch.Tensor,
        field_scores: torch.Tensor,
        objectness_logits: torch.Tensor,
    ) -> Dict[str, float]:
        field_hits = [0, 0, 0]
        field_totals = [0, 0, 0]
        tuple_hits = 0
        tuple_total = 0
        matched_hits = 0
        matched_total = 0
        gold_costs = []
        pred_costs = []
        for b in range(batch["input_ids"].size(0)):
            target_indices, _ = self._target_candidate_indices(batch, candidate_positions, b)
            cost = self._candidate_cost_matrix(field_scores, objectness_logits, target_indices, b)
            pairs = _assignment_from_cost(cost) if len(target_indices) > 0 else []
            for true_idx, pred_idx in pairs:
                matched_total += 1
                if float(cost[true_idx, pred_idx].detach().item()) < 1.0e-4:
                    matched_hits += 1
                gold_costs.append(float(cost[true_idx, pred_idx].detach().item()))
            for out_idx, indices in enumerate(target_indices[: self.max_slots]):
                tuple_total += 1
                tuple_ok = True
                for field, target_idx in enumerate(indices):
                    pred_idx = int(predicted_candidate_indices[b, out_idx, field].item())
                    ok = pred_idx == int(target_idx)
                    field_hits[field] += int(ok)
                    field_totals[field] += 1
                    tuple_ok = tuple_ok and ok
                    pred_costs.append(float((-F.log_softmax(field_scores[b, out_idx, field, :], dim=-1)[target_idx]).detach().item()))
                tuple_hits += int(tuple_ok)
        return {
            "key_candidate_accuracy": field_hits[0] / max(field_totals[0], 1),
            "condition_candidate_accuracy": (field_hits[2] / max(field_totals[2], 1)) if self.has_condition else 1.0,
            "value_candidate_accuracy": field_hits[1] / max(field_totals[1], 1),
            "tuple_accuracy": tuple_hits / max(tuple_total, 1),
            "matched_tuple_accuracy": matched_hits / max(matched_total, 1),
            "mean_hungarian_gold_cost": sum(gold_costs) / max(len(gold_costs), 1),
            "mean_hungarian_pred_cost": sum(pred_costs) / max(len(pred_costs), 1),
        }

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: int,
        candidate_mode: str,
        assembly_eval_mode: str = "independent_field_heads_current",
    ) -> Dict[str, torch.Tensor]:
        field_scores, objectness_logits, candidate_positions, candidate_mask, _ = self.forward_with_candidates(
            batch,
            candidate_k,
            candidate_mode,
        )
        predicted_candidate_indices = field_scores.argmax(dim=-1)
        predicted_positions = torch.gather(
            candidate_positions[:, None, :, :].expand(-1, self.max_slots, -1, -1),
            dim=-1,
            index=predicted_candidate_indices[..., None],
        ).squeeze(-1)
        objectness_probs = torch.sigmoid(objectness_logits)
        active = objectness_probs > self.objectness_threshold

        if assembly_eval_mode in {"no_objectness_true_count", "no_hungarian_canonical_debug"}:
            active = torch.zeros_like(active)
            for b in range(batch["input_ids"].size(0)):
                true_count = int(batch["memory_mask"][b].sum().item())
                active[b, : min(true_count, self.max_slots)] = True
        gold_field_modes = {
            "gold_key_only": [0],
            "gold_cond_only": [2],
            "gold_value_only": [1],
            "gold_key_cond": [0, 2],
            "gold_all_fields": [0, 1, 2],
        }
        if assembly_eval_mode in gold_field_modes:
            active = torch.zeros_like(active)
            for b in range(batch["input_ids"].size(0)):
                true_count = int(batch["memory_mask"][b].sum().item())
                active[b, : min(true_count, self.max_slots)] = True
            predicted_positions = self._with_gold_fields(predicted_positions, batch, gold_field_modes[assembly_eval_mode])

        if candidate_mode == "learned_candidates_oracle_assembly":
            active = torch.zeros_like(active)
            predicted_positions = torch.full_like(predicted_positions, -1)
            for b in range(batch["input_ids"].size(0)):
                target_indices, source_slots = self._target_candidate_indices(batch, candidate_positions, b)
                for out_idx, (_, slot) in enumerate(zip(target_indices, source_slots)):
                    if out_idx >= self.max_slots:
                        break
                    active[b, out_idx] = True
                    predicted_positions[b, out_idx, 0] = batch["memory_token_positions"][b, slot, 0]
                    predicted_positions[b, out_idx, 1] = batch["memory_token_positions"][b, slot, 1]
                    if self.has_condition:
                        predicted_positions[b, out_idx, 2] = batch["memory_condition_positions"][b, slot]

        rewritten = clone_batch(batch)
        rewritten["memory_token_positions"] = torch.stack([predicted_positions[:, :, 0], predicted_positions[:, :, 1]], dim=-1)
        if self.has_condition:
            rewritten["memory_condition_positions"] = predicted_positions[:, :, 2]
        else:
            rewritten["memory_condition_positions"] = torch.full_like(predicted_positions[:, :, 0], -1)
        starts = torch.minimum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        ends = torch.maximum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        if self.has_condition:
            starts = torch.minimum(starts, rewritten["memory_condition_positions"])
            ends = torch.maximum(ends, rewritten["memory_condition_positions"])
        rewritten["memory_spans"] = torch.stack([starts, ends], dim=-1)
        rewritten["memory_mask"] = active
        rewritten["objectness_probs"] = objectness_probs
        rewritten["candidate_positions"] = candidate_positions
        rewritten["candidate_mask"] = candidate_mask
        rewritten["candidate_k"] = torch.tensor(candidate_k, device=batch["input_ids"].device)
        rewritten["assembler_debug_metrics"] = self._debug_metric_dict(
            batch,
            candidate_positions,
            predicted_candidate_indices,
            field_scores,
            objectness_logits,
        )
        return rewritten

    @torch.no_grad()
    def debug_examples(self, batch: Dict[str, torch.Tensor], candidate_k: int, limit: int) -> List[Dict[str, object]]:
        field_scores, objectness_logits, candidate_positions, candidate_mask, _ = self.forward_with_candidates(
            batch,
            candidate_k,
            "oracle_candidates_gold_only",
        )
        predicted_candidate_indices = field_scores.argmax(dim=-1)
        examples = []
        input_ids = batch["input_ids"]
        for b in range(min(batch["input_ids"].size(0), limit)):
            target_indices, source_slots = self._target_candidate_indices(batch, candidate_positions, b)
            cost = self._candidate_cost_matrix(field_scores, objectness_logits, target_indices, b)
            pairs = _assignment_from_cost(cost) if len(target_indices) > 0 else []
            true_slots = []
            for slot in source_slots:
                true_slots.append(
                    {
                        "slot": slot,
                        "key_pos": int(batch["memory_token_positions"][b, slot, 0].item()),
                        "key_token": int(input_ids[b, batch["memory_token_positions"][b, slot, 0]].item()),
                        "value_pos": int(batch["memory_token_positions"][b, slot, 1].item()),
                        "value_token": int(input_ids[b, batch["memory_token_positions"][b, slot, 1]].item()),
                        "condition_pos": int(batch["memory_condition_positions"][b, slot].item()) if self.has_condition else -1,
                        "condition_token": int(input_ids[b, batch["memory_condition_positions"][b, slot]].item()) if self.has_condition else -1,
                    }
                )
            candidates = {}
            names = ["key", "value", "condition"] if self.has_condition else ["key", "value"]
            for field, name in enumerate(names):
                candidates[name] = [
                    {"index": idx, "pos": int(pos.item()), "token": int(input_ids[b, pos].item())}
                    for idx, pos in enumerate(candidate_positions[b, field])
                    if bool(candidate_mask[b, field, idx].item())
                ]
            predictions = []
            for slot in range(self.max_slots):
                pred_indices = [int(x.item()) for x in predicted_candidate_indices[b, slot, : self.num_fields]]
                predictions.append(
                    {
                        "pred_slot": slot,
                        "candidate_indices": pred_indices,
                        "objectness": float(torch.sigmoid(objectness_logits[b, slot]).item()),
                    }
                )
            examples.append(
                {
                    "sample": b,
                    "true_slots": true_slots,
                    "candidates": candidates,
                    "target_candidate_indices": target_indices,
                    "predictions": predictions,
                    "hungarian_cost_matrix": cost.detach().cpu().tolist(),
                    "hungarian_assignment_true_to_pred": pairs,
                }
            )
        return examples


class SPNTupleAssembler(nn.Module):
    """Whole-tuple scorer over candidate fields for tuple assembly sanity checks."""

    def __init__(
        self,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.proposer = CandidateFieldProposer(seq_len, has_condition, extractor_dim, hidden, layers, dropout)
        in_dim = extractor_dim * (6 if has_condition else 3)
        self.scorer = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def _tuple_features(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key = states[:, 0]
        value = states[:, 1]
        if self.has_condition:
            cond = states[:, 2]
            features = torch.cat(
                [
                    key[:, :, None, None, :].expand(-1, -1, cond.size(1), value.size(1), -1),
                    cond[:, None, :, None, :].expand(-1, key.size(1), -1, value.size(1), -1),
                    value[:, None, None, :, :].expand(-1, key.size(1), cond.size(1), -1, -1),
                    (key[:, :, None, None, :] * cond[:, None, :, None, :]).expand(-1, -1, -1, value.size(1), -1),
                    (key[:, :, None, None, :] * value[:, None, None, :, :]).expand(-1, -1, cond.size(1), -1, -1),
                    (cond[:, None, :, None, :] * value[:, None, None, :, :]).expand(-1, key.size(1), -1, -1, -1),
                ],
                dim=-1,
            )
            return features.flatten(1, 3), torch.tensor([key.size(1), cond.size(1), value.size(1)], device=states.device)
        features = torch.cat(
            [
                key[:, :, None, :].expand(-1, -1, value.size(1), -1),
                value[:, None, :, :].expand(-1, key.size(1), -1, -1),
                key[:, :, None, :] * value[:, None, :, :],
            ],
            dim=-1,
        )
        return features.flatten(1, 2), torch.tensor([key.size(1), value.size(1)], device=states.device)

    def _target_tuple_indices(self, assembler: WriterV3CandidateAssembler, batch: Dict[str, torch.Tensor], candidate_positions: torch.Tensor, b: int) -> List[int]:
        target_indices, _ = assembler._target_candidate_indices(batch, candidate_positions, b)
        dims = [candidate_positions.size(2)] * assembler.num_fields
        out = []
        for indices in target_indices:
            if assembler.has_condition:
                out.append(indices[0] * dims[1] * dims[2] + indices[2] * dims[2] + indices[1])
            else:
                out.append(indices[0] * dims[1] + indices[1])
        return out

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: int,
        candidate_loss_weight: float = 1.0,
        tuple_loss_weight: float = 1.0,
        rank_loss_weight: float = 0.5,
    ) -> torch.Tensor:
        candidate_loss = self.proposer.loss(batch)
        candidate_positions, _, _, encoded = self.proposer.candidate_pools(batch, candidate_k, "oracle_candidates_gold_only")
        helper = WriterV3CandidateAssembler(candidate_k, encoded.size(1), self.has_condition).to(encoded.device)
        states = helper._candidate_states(encoded, candidate_positions)
        features, _ = self._tuple_features(states)
        scores = self.scorer(features).squeeze(-1)
        losses = []
        for b in range(batch["input_ids"].size(0)):
            target_tuple_indices = self._target_tuple_indices(helper, batch, candidate_positions, b)
            for target_idx in target_tuple_indices:
                losses.append(F.cross_entropy(scores[b][None, :], torch.tensor([target_idx], device=scores.device)))
        tuple_loss = scores.new_zeros(()) if not losses else torch.stack(losses).mean()
        return candidate_loss_weight * candidate_loss + tuple_loss

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, torch.Tensor], candidate_k: int) -> Dict[str, torch.Tensor]:
        candidate_positions, candidate_mask, _, encoded = self.proposer.candidate_pools(batch, candidate_k, "oracle_candidates_gold_only")
        helper = WriterV3CandidateAssembler(candidate_k, encoded.size(1), self.has_condition).to(encoded.device)
        states = helper._candidate_states(encoded, candidate_positions)
        features, dims = self._tuple_features(states)
        scores = self.scorer(features).squeeze(-1)
        max_slots = int(batch["memory_mask"].size(1))
        predicted_positions = torch.full((batch["input_ids"].size(0), max_slots, helper.num_fields), -1, device=batch["input_ids"].device, dtype=torch.long)
        active = torch.zeros((batch["input_ids"].size(0), max_slots), device=batch["input_ids"].device, dtype=torch.bool)
        for b in range(batch["input_ids"].size(0)):
            true_count = min(int(batch["memory_mask"][b].sum().item()), max_slots)
            if true_count == 0:
                continue
            top_indices = torch.topk(scores[b], k=true_count).indices
            for out_idx, flat_idx_tensor in enumerate(top_indices):
                flat_idx = int(flat_idx_tensor.item())
                if self.has_condition:
                    k_count, c_count, v_count = [int(x.item()) for x in dims]
                    key_idx = flat_idx // (c_count * v_count)
                    rem = flat_idx % (c_count * v_count)
                    cond_idx = rem // v_count
                    value_idx = rem % v_count
                    predicted_positions[b, out_idx, 0] = candidate_positions[b, 0, key_idx]
                    predicted_positions[b, out_idx, 1] = candidate_positions[b, 1, value_idx]
                    predicted_positions[b, out_idx, 2] = candidate_positions[b, 2, cond_idx]
                else:
                    _, v_count = [int(x.item()) for x in dims]
                    key_idx = flat_idx // v_count
                    value_idx = flat_idx % v_count
                    predicted_positions[b, out_idx, 0] = candidate_positions[b, 0, key_idx]
                    predicted_positions[b, out_idx, 1] = candidate_positions[b, 1, value_idx]
                active[b, out_idx] = True
        rewritten = clone_batch(batch)
        rewritten["memory_token_positions"] = torch.stack([predicted_positions[:, :, 0], predicted_positions[:, :, 1]], dim=-1)
        if self.has_condition:
            rewritten["memory_condition_positions"] = predicted_positions[:, :, 2]
        else:
            rewritten["memory_condition_positions"] = torch.full_like(predicted_positions[:, :, 0], -1)
        starts = torch.minimum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        ends = torch.maximum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        if self.has_condition:
            starts = torch.minimum(starts, rewritten["memory_condition_positions"])
            ends = torch.maximum(ends, rewritten["memory_condition_positions"])
        rewritten["memory_spans"] = torch.stack([starts, ends], dim=-1)
        rewritten["memory_mask"] = active
        rewritten["candidate_positions"] = candidate_positions
        rewritten["candidate_mask"] = candidate_mask
        rewritten["assembler_debug_metrics"] = {
            "spn_tuple_exact": 0.0,
            "spn_tuple_accuracy": 0.0,
        }
        return rewritten


class ContextualTupleEdgeScorer(nn.Module):
    """Context-aware tuple scorer for Writer v4.

    V3/SPN scored candidate fields mostly from isolated field embeddings. This
    scorer adds token context, field positions, relative order, local windows,
    and pooled text between the fields. It still uses synthetic typed candidate
    pools and decodes with true slot count for the first diagnostic so tuple
    scoring is isolated from cardinality calibration.
    """

    def __init__(
        self,
        max_slots: int,
        seq_len: int,
        has_condition: bool,
        extractor_dim: int = 64,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.0,
        tuple_threshold: float = 0.5,
        rank_margin: float = 1.0,
        condition_proposer_variant: str = "baseline",
        simplified_aux_weight: float = 0.0,
        guideline_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.seq_len = seq_len
        self.has_condition = has_condition
        self.num_fields = 3 if has_condition else 2
        self.tuple_threshold = tuple_threshold
        self.rank_margin = rank_margin
        if condition_proposer_variant == "coexisting_full":
            self.proposer = CoexistingCandidateProposerV2(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                span_fields=True,
                guideline_fields=True,
                simplified_aux_weight=simplified_aux_weight,
                guideline_loss_weight=guideline_loss_weight,
            )
        elif condition_proposer_variant == "baseline" or not has_condition:
            self.proposer = CandidateFieldProposer(seq_len, has_condition, extractor_dim, hidden, layers, dropout)
        elif condition_proposer_variant == "span":
            self.proposer = ConditionCandidateProposerV2(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                span_condition=True,
            )
        elif condition_proposer_variant == "guideline":
            self.proposer = ConditionCandidateProposerV2(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                guideline_condition=True,
                guideline_loss_weight=guideline_loss_weight,
            )
        elif condition_proposer_variant == "simplified":
            self.proposer = ConditionCandidateProposerV2(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                simplified_aux_weight=simplified_aux_weight,
            )
        elif condition_proposer_variant == "full":
            self.proposer = ConditionCandidateProposerV2(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                span_condition=True,
                guideline_condition=True,
                simplified_aux_weight=simplified_aux_weight,
                guideline_loss_weight=guideline_loss_weight,
            )
        elif condition_proposer_variant == "condition_v3":
            self.proposer = ConditionCandidateProposerV3(
                seq_len,
                has_condition,
                extractor_dim,
                hidden,
                layers,
                dropout,
                span_condition=True,
                guideline_condition=True,
                simplified_aux_weight=simplified_aux_weight,
                guideline_loss_weight=guideline_loss_weight,
            )
        else:
            raise ValueError(f"unknown condition_proposer_variant: {condition_proposer_variant}")
        self.token_emb = nn.Embedding(VOCAB_SIZE, extractor_dim)
        self.pos_emb = nn.Embedding(seq_len, extractor_dim)
        modules = []
        current = extractor_dim * 5
        for _ in range(max(layers - 1, 0)):
            modules.append(nn.Linear(current, hidden))
            modules.append(nn.Tanh())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            current = hidden
        modules.append(nn.Linear(current, extractor_dim))
        self.context = nn.Sequential(*modules)
        feature_dim = extractor_dim * (13 if has_condition else 7) + (9 if has_condition else 4)
        scorer = []
        current = feature_dim
        for _ in range(max(layers - 1, 0)):
            scorer.append(nn.Linear(current, hidden))
            scorer.append(nn.Tanh())
            if dropout > 0:
                scorer.append(nn.Dropout(dropout))
            current = hidden
        scorer.append(nn.Linear(current, 1))
        self.scorer = nn.Sequential(*scorer)
        self._last_candidate_debug: Dict[str, float] = {}
        self._last_candidate_logits: torch.Tensor | None = None
        self._last_profile: Dict[str, float] = {}

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        windows = [_shift_with_zero(x, shift) for shift in (-2, -1, 0, 1, 2)]
        return self.context(torch.cat(windows, dim=-1))

    def candidate_pools(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        simple_modes = {
            "oracle_candidates",
            "oracle_candidates_plus_noise",
            "oracle_candidates_gold_only",
            "learned_candidates",
        }
        if mode in simple_modes:
            positions, mask, logits, _ = self.proposer.candidate_pools(batch, candidate_k, mode)
            self._last_candidate_logits = logits
            encoded = self.encode(batch["input_ids"])
            return positions, mask, encoded

        with torch.no_grad():
            logits, _ = self.proposer.forward(batch)
            self._last_candidate_logits = logits
            masks = _pre_query_field_masks(batch["input_ids"], self.has_condition)
            bsz = batch["input_ids"].size(0)
            field_ks = _candidate_k_by_field(candidate_k, self.num_fields)
            max_candidate_k = max(field_ks)
            positions = torch.full((bsz, self.num_fields, max_candidate_k), -1, device=batch["input_ids"].device, dtype=torch.long)
            mask = torch.zeros((bsz, self.num_fields, max_candidate_k), device=batch["input_ids"].device, dtype=torch.bool)
            oracle_fields_by_mode = {
                "learned_candidates_plus_oracle_missing": set(),
                "oracle_candidates_plus_learned_noise": {0, 1, 2},
                "oracle_key_candidates": {0},
                "oracle_cond_candidates": {2},
                "oracle_value_candidates": {1},
                "oracle_key_cond_candidates": {0, 2},
                "oracle_key_value_candidates": {0, 1},
                "oracle_cond_value_candidates": {1, 2},
            }
            oracle_fields = oracle_fields_by_mode.get(mode)
            if oracle_fields is None:
                raise ValueError(f"unknown contextual candidate mode: {mode}")
            for b in range(bsz):
                gold_fields = _gold_field_position_lists(batch, self.has_condition, b)
                for field in range(self.num_fields):
                    field_k = field_ks[field]
                    field_logits = logits[b, field].clone()
                    valid_positions = torch.nonzero(masks[b, field], as_tuple=False).flatten()
                    selected: List[int] = []
                    gold = [pos for pos in gold_fields[field] if pos >= 0]
                    use_oracle_for_field = field in oracle_fields
                    if use_oracle_for_field:
                        for pos in gold:
                            if pos not in selected and len(selected) < field_k:
                                selected.append(pos)
                    if mode == "learned_candidates_plus_oracle_missing" or not use_oracle_for_field:
                        top = torch.topk(field_logits, k=min(field_k, int(valid_positions.numel()))).indices.tolist()
                        selected.extend(int(pos) for pos in top if int(pos) not in selected)
                    else:
                        for pos in gold:
                            field_logits[pos] = -1.0e9
                        fill_count = field_k - len(selected)
                        if fill_count > 0:
                            top = torch.topk(field_logits, k=min(fill_count, int(valid_positions.numel()))).indices.tolist()
                            selected.extend(int(pos) for pos in top if int(pos) not in selected)
                    if mode == "learned_candidates_plus_oracle_missing":
                        for pos in gold:
                            if pos in selected:
                                continue
                            if len(selected) < field_k:
                                selected.append(pos)
                                continue
                            replace_idx = next(
                                (idx for idx in range(len(selected) - 1, -1, -1) if selected[idx] not in gold),
                                len(selected) - 1,
                            )
                            selected[replace_idx] = pos
                    selected = selected[:field_k]
                    if selected:
                        positions[b, field, : len(selected)] = torch.tensor(selected, device=positions.device, dtype=torch.long)
                        mask[b, field, : len(selected)] = True
        encoded = self.encode(batch["input_ids"])
        return positions, mask, encoded

    def _candidate_states(self, encoded: torch.Tensor, candidate_positions: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = encoded.shape
        safe = candidate_positions.clamp_min(0).clamp_max(seq_len - 1)
        expanded_encoded = encoded[:, None, :, :].expand(-1, self.num_fields, -1, -1)
        gather_index = safe[..., None].expand(-1, -1, -1, dim)
        states = torch.gather(expanded_encoded, dim=2, index=gather_index)
        return torch.where(candidate_positions[..., None] >= 0, states, torch.zeros_like(states))

    def _enumerate_tuple_positions(
        self,
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_pos = candidate_positions[:, 0]
        value_pos = candidate_positions[:, 1]
        key_mask = candidate_mask[:, 0]
        value_mask = candidate_mask[:, 1]
        if self.has_condition:
            cond_pos = candidate_positions[:, 2]
            cond_mask = candidate_mask[:, 2]
            k_count = key_pos.size(1)
            c_count = cond_pos.size(1)
            v_count = value_pos.size(1)
            key_grid = key_pos[:, :, None, None].expand(-1, -1, c_count, v_count)
            cond_grid = cond_pos[:, None, :, None].expand(-1, k_count, -1, v_count)
            value_grid = value_pos[:, None, None, :].expand(-1, k_count, c_count, -1)
            mask = (
                key_mask[:, :, None, None]
                & cond_mask[:, None, :, None]
                & value_mask[:, None, None, :]
            )
            positions = torch.stack([key_grid, value_grid, cond_grid], dim=-1).flatten(1, 3)
            return positions, mask.flatten(1, 3)
        k_count = key_pos.size(1)
        v_count = value_pos.size(1)
        key_grid = key_pos[:, :, None].expand(-1, -1, v_count)
        value_grid = value_pos[:, None, :].expand(-1, k_count, -1)
        positions = torch.stack([key_grid, value_grid], dim=-1).flatten(1, 2)
        return positions, (key_mask[:, :, None] & value_mask[:, None, :]).flatten(1, 2)

    def _gather_states(self, encoded: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, _, dim = encoded.shape
        safe = positions.clamp_min(0).clamp_max(encoded.size(1) - 1)
        batch_index = torch.arange(bsz, device=encoded.device).view(bsz, 1, 1).expand_as(safe)
        states = encoded[batch_index, safe]
        return torch.where(positions[..., None] >= 0, states, torch.zeros_like(states))

    def _span_mean(self, encoded: torch.Tensor, tuple_positions: torch.Tensor) -> torch.Tensor:
        valid_positions = tuple_positions.clamp_min(0)
        start = valid_positions.min(dim=-1).values
        end = valid_positions.max(dim=-1).values
        prefix = torch.cat([encoded.new_zeros(encoded.size(0), 1, encoded.size(2)), encoded.cumsum(dim=1)], dim=1)
        bsz = encoded.size(0)
        batch_index = torch.arange(bsz, device=encoded.device)[:, None].expand_as(start)
        span_sum = prefix[batch_index, end + 1] - prefix[batch_index, start]
        length = (end - start + 1).clamp_min(1).to(encoded.dtype)
        return span_sum / length[..., None]

    def _precompute_local_mean(self, encoded: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Precompute 5-token window mean for every sequence position. [B, seq_len, D]"""
        bsz, seq_len, dim = encoded.shape
        query_positions = first_positions(input_ids, QUERY)
        arange = torch.arange(seq_len, device=encoded.device)
        pre_query = arange[None, :] < query_positions[:, None]
        src_enc = encoded * pre_query.unsqueeze(-1).to(encoded.dtype)
        src_valid = pre_query.to(encoded.dtype)
        total = encoded.new_zeros(bsz, seq_len, dim)
        count = encoded.new_zeros(bsz, seq_len)
        for offset in (-2, -1, 0, 1, 2):
            if offset == 0:
                total.add_(src_enc)
                count.add_(src_valid)
            elif offset > 0:
                total[:, :seq_len - offset].add_(src_enc[:, offset:])
                count[:, :seq_len - offset].add_(src_valid[:, offset:])
            else:
                abs_off = -offset
                total[:, abs_off:].add_(src_enc[:, :seq_len - abs_off])
                count[:, abs_off:].add_(src_valid[:, :seq_len - abs_off])
        return total / count.clamp_min(1).unsqueeze(-1)

    def _local_mean(self, encoded: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        bsz = encoded.size(0)
        local_mean_seq = self._precompute_local_mean(encoded, input_ids)
        safe = positions.clamp_min(0).clamp_max(encoded.size(1) - 1)
        batch_index = torch.arange(bsz, device=encoded.device)[:, None, None].expand_as(safe)
        result = local_mean_seq[batch_index, safe]
        return torch.where(positions[..., None] >= 0, result, torch.zeros_like(result))

    def _tuple_features(
        self,
        batch: Dict[str, torch.Tensor],
        encoded: torch.Tensor,
        tuple_positions: torch.Tensor,
    ) -> torch.Tensor:
        field_states = self._gather_states(encoded, tuple_positions)
        key = field_states[:, :, 0]
        value = field_states[:, :, 1]
        between = self._span_mean(encoded, tuple_positions)
        local = self._local_mean(encoded, tuple_positions, batch["input_ids"])
        key_local = local[:, :, 0]
        value_local = local[:, :, 1]
        seq_scale = float(max(self.seq_len, 1))

        if self.has_condition:
            cond = field_states[:, :, 2]
            cond_local = local[:, :, 2]
            key_pos = tuple_positions[:, :, 0].to(encoded.dtype)
            value_pos = tuple_positions[:, :, 1].to(encoded.dtype)
            cond_pos = tuple_positions[:, :, 2].to(encoded.dtype)
            pos_features = torch.stack(
                [
                    key_pos / seq_scale,
                    cond_pos / seq_scale,
                    value_pos / seq_scale,
                    (key_pos - cond_pos).abs() / seq_scale,
                    (cond_pos - value_pos).abs() / seq_scale,
                    (key_pos - value_pos).abs() / seq_scale,
                    (key_pos < cond_pos).to(encoded.dtype),
                    (cond_pos < value_pos).to(encoded.dtype),
                    (key_pos < value_pos).to(encoded.dtype),
                ],
                dim=-1,
            )
            return torch.cat(
                [
                    key,
                    cond,
                    value,
                    key * cond,
                    key * value,
                    cond * value,
                    (key - cond).abs(),
                    (key - value).abs(),
                    (cond - value).abs(),
                    pos_features,
                    between,
                    key_local,
                    cond_local,
                    value_local,
                ],
                dim=-1,
            )

        key_pos = tuple_positions[:, :, 0].to(encoded.dtype)
        value_pos = tuple_positions[:, :, 1].to(encoded.dtype)
        pos_features = torch.stack(
            [
                key_pos / seq_scale,
                value_pos / seq_scale,
                (key_pos - value_pos).abs() / seq_scale,
                (key_pos < value_pos).to(encoded.dtype),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                key,
                value,
                key * value,
                (key - value).abs(),
                pos_features,
                between,
                key_local,
                value_local,
            ],
            dim=-1,
        )

    def score_tuples(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        candidate_mode: str,
        tuple_pruning: str = "none",
        pair_beam_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t0 = time.perf_counter()
        candidate_positions, candidate_mask, encoded = self.candidate_pools(batch, candidate_k, candidate_mode)
        t1 = time.perf_counter()
        self._last_candidate_debug = dict(getattr(self.proposer, "last_candidate_debug", {}))
        if tuple_pruning == "pair_beam" and self.has_condition:
            tuple_positions, tuple_mask = self._enumerate_pair_beam_tuple_positions(
                batch,
                candidate_positions,
                candidate_mask,
                pair_beam_size,
            )
        elif tuple_pruning == "none":
            tuple_positions, tuple_mask = self._enumerate_tuple_positions(candidate_positions, candidate_mask)
        else:
            raise ValueError(f"unknown tuple_pruning: {tuple_pruning}")
        features = self._tuple_features(batch, encoded, tuple_positions)
        scores = self.scorer(features).squeeze(-1)
        scores = scores.masked_fill(~tuple_mask, -1.0e9)
        t2 = time.perf_counter()
        self._last_profile = {
            "candidate_proposer_time": t1 - t0,
            "tuple_scorer_time": t2 - t1,
            "tuple_candidates_scored": float(tuple_positions.size(1)),
            "tuple_pruning_pair_beam_size": float(pair_beam_size if tuple_pruning == "pair_beam" else 0),
        }
        return scores, tuple_positions, tuple_mask, candidate_positions, candidate_mask

    def _enumerate_pair_beam_tuple_positions(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        pair_beam_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = candidate_positions.size(0)
        candidate_k = candidate_positions.size(2)
        pair_beam_size = max(1, min(int(pair_beam_size), candidate_k * candidate_k))
        max_tuples = pair_beam_size * candidate_k
        tuple_positions = torch.full((bsz, max_tuples, self.num_fields), -1, device=candidate_positions.device, dtype=torch.long)
        tuple_mask = torch.zeros((bsz, max_tuples), device=candidate_positions.device, dtype=torch.bool)
        logits = self._last_candidate_logits
        for b in range(bsz):
            key_valid = torch.nonzero(candidate_mask[b, 0], as_tuple=False).flatten()
            value_valid = torch.nonzero(candidate_mask[b, 1], as_tuple=False).flatten()
            cond_valid = torch.nonzero(candidate_mask[b, 2], as_tuple=False).flatten()
            if key_valid.numel() == 0 or value_valid.numel() == 0 or cond_valid.numel() == 0:
                continue
            pairs = []
            for key_idx_tensor in key_valid:
                key_idx = int(key_idx_tensor.item())
                key_pos = int(candidate_positions[b, 0, key_idx].item())
                for cond_idx_tensor in cond_valid:
                    cond_idx = int(cond_idx_tensor.item())
                    cond_pos = int(candidate_positions[b, 2, cond_idx].item())
                    distance = abs(key_pos - cond_pos) / float(max(self.seq_len, 1))
                    score = -4.0 * distance
                    if logits is not None:
                        score += float(logits[b, 0, key_pos].detach().item())
                        score += float(logits[b, 2, cond_pos].detach().item())
                    pairs.append((score, key_idx, cond_idx))
            pairs.sort(key=lambda item: item[0], reverse=True)
            out = 0
            for _, key_idx, cond_idx in pairs[:pair_beam_size]:
                for value_idx_tensor in value_valid:
                    if out >= max_tuples:
                        break
                    value_idx = int(value_idx_tensor.item())
                    tuple_positions[b, out, 0] = candidate_positions[b, 0, key_idx]
                    tuple_positions[b, out, 1] = candidate_positions[b, 1, value_idx]
                    tuple_positions[b, out, 2] = candidate_positions[b, 2, cond_idx]
                    tuple_mask[b, out] = True
                    out += 1
        return tuple_positions, tuple_mask

    def _target_candidate_indices(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_positions: torch.Tensor,
        b: int,
    ) -> Tuple[List[List[int]], List[int]]:
        valid = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
        target_indices: List[List[int]] = []
        source_slots: List[int] = []
        for slot_tensor in valid:
            slot = int(slot_tensor.item())
            targets = [
                int(batch["memory_token_positions"][b, slot, 0].item()),
                int(batch["memory_token_positions"][b, slot, 1].item()),
            ]
            if self.has_condition:
                targets.append(int(batch["memory_condition_positions"][b, slot].item()))
            indices = []
            missing = False
            for field, target in enumerate(targets):
                matches = torch.nonzero(candidate_positions[b, field] == target, as_tuple=False).flatten()
                if matches.numel() == 0:
                    missing = True
                    break
                indices.append(int(matches[0].item()))
            if not missing:
                target_indices.append(indices)
                source_slots.append(slot)
        return target_indices, source_slots

    def _target_tuple_indices(self, batch: Dict[str, torch.Tensor], candidate_positions: torch.Tensor, b: int) -> Tuple[List[int], List[int]]:
        target_indices, source_slots = self._target_candidate_indices(batch, candidate_positions, b)
        k_count = candidate_positions.size(2)
        out = []
        for indices in target_indices:
            if self.has_condition:
                out.append(indices[0] * k_count * k_count + indices[2] * k_count + indices[1])
            else:
                out.append(indices[0] * k_count + indices[1])
        return out, source_slots

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        candidate_loss_weight: float = 1.0,
        tuple_loss_weight: float = 1.0,
        rank_loss_weight: float = 0.5,
        tuple_pruning: str = "none",
        pair_beam_size: int = 8,
    ) -> torch.Tensor:
        candidate_loss = self.proposer.loss(batch)
        scores, _, tuple_mask, candidate_positions, _ = self.score_tuples(
            batch,
            candidate_k,
            "oracle_candidates_plus_noise",
            tuple_pruning=tuple_pruning,
            pair_beam_size=pair_beam_size,
        )
        labels = torch.zeros_like(scores)
        for b in range(scores.size(0)):
            target_tuple_indices, _ = self._target_tuple_indices(batch, candidate_positions, b)
            for target_idx in target_tuple_indices:
                if 0 <= target_idx < labels.size(1) and bool(tuple_mask[b, target_idx].item()):
                    labels[b, target_idx] = 1.0

        valid_scores = scores[tuple_mask]
        valid_labels = labels[tuple_mask]
        if valid_scores.numel() == 0 or not bool(valid_labels.any().item()):
            tuple_loss = scores.new_zeros(())
        else:
            positives = valid_labels.sum().clamp_min(1.0)
            negatives = (valid_labels.numel() - valid_labels.sum()).clamp_min(1.0)
            pos_weight = (negatives / positives).clamp(max=50.0)
            bce = F.binary_cross_entropy_with_logits(valid_scores, valid_labels, pos_weight=pos_weight)
            rank_losses = []
            for b in range(scores.size(0)):
                pos_scores = scores[b][tuple_mask[b] & (labels[b] > 0.5)]
                neg_scores = scores[b][tuple_mask[b] & (labels[b] < 0.5)]
                if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                    continue
                hard_neg = neg_scores.topk(k=min(32, neg_scores.numel())).values
                rank_losses.append(F.relu(self.rank_margin - pos_scores[:, None] + hard_neg[None, :]).mean())
            rank = scores.new_zeros(()) if not rank_losses else torch.stack(rank_losses).mean()
            tuple_loss = tuple_loss_weight * bce + rank_loss_weight * rank
        return candidate_loss_weight * candidate_loss + tuple_loss

    def _gold_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rewritten = clone_batch(batch)
        rewritten["assembler_debug_metrics"] = {
            "contextual_tuple_exact": 1.0,
            "contextual_tuple_slot_f1": 1.0,
            "contextual_tuple_all_slots_exact": 1.0,
            "top_true_count_exact": 1.0,
        }
        return rewritten

    def _build_rewritten(
        self,
        batch: Dict[str, torch.Tensor],
        predicted_positions: torch.Tensor,
        active: torch.Tensor,
        candidate_positions: torch.Tensor,
        candidate_mask: torch.Tensor,
        debug_metrics: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        rewritten = clone_batch(batch)
        rewritten["memory_token_positions"] = torch.stack([predicted_positions[:, :, 0], predicted_positions[:, :, 1]], dim=-1)
        if self.has_condition:
            rewritten["memory_condition_positions"] = predicted_positions[:, :, 2]
        else:
            rewritten["memory_condition_positions"] = torch.full_like(predicted_positions[:, :, 0], -1)
        starts = torch.minimum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        ends = torch.maximum(rewritten["memory_token_positions"][:, :, 0], rewritten["memory_token_positions"][:, :, 1])
        if self.has_condition:
            starts = torch.minimum(starts, rewritten["memory_condition_positions"])
            ends = torch.maximum(ends, rewritten["memory_condition_positions"])
        rewritten["memory_spans"] = torch.stack([starts, ends], dim=-1)
        rewritten["memory_mask"] = active
        rewritten["candidate_positions"] = candidate_positions
        rewritten["candidate_mask"] = candidate_mask
        rewritten["assembler_debug_metrics"] = debug_metrics
        return rewritten

    def _select_top_true_count(
        self,
        scores: torch.Tensor,
        tuple_positions: torch.Tensor,
        tuple_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = batch["input_ids"].size(0)
        predicted = torch.full((bsz, self.max_slots, self.num_fields), -1, device=scores.device, dtype=torch.long)
        active = torch.zeros((bsz, self.max_slots), device=scores.device, dtype=torch.bool)
        for b in range(bsz):
            true_count = min(int(batch["memory_mask"][b].sum().item()), self.max_slots)
            if true_count == 0:
                continue
            valid_indices = torch.nonzero(tuple_mask[b], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            ranked = valid_indices[torch.argsort(scores[b, valid_indices], descending=True)]
            seen = set()
            out_idx = 0
            for flat_idx_tensor in ranked:
                flat_idx = int(flat_idx_tensor.item())
                fields = tuple(int(x) for x in tuple_positions[b, flat_idx, : self.num_fields].tolist())
                if fields in seen:
                    continue
                seen.add(fields)
                predicted[b, out_idx, : self.num_fields] = tuple_positions[b, flat_idx, : self.num_fields]
                active[b, out_idx] = True
                out_idx += 1
                if out_idx >= true_count:
                    break
        return predicted, active

    def _select_threshold(
        self,
        scores: torch.Tensor,
        tuple_positions: torch.Tensor,
        tuple_mask: torch.Tensor,
        threshold: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = scores.size(0)
        predicted = torch.full((bsz, self.max_slots, self.num_fields), -1, device=scores.device, dtype=torch.long)
        active = torch.zeros((bsz, self.max_slots), device=scores.device, dtype=torch.bool)
        probs = torch.sigmoid(scores)
        for b in range(bsz):
            valid_indices = torch.nonzero(tuple_mask[b] & (probs[b] > threshold), as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            ranked = valid_indices[torch.argsort(scores[b, valid_indices], descending=True)]
            for out_idx, flat_idx_tensor in enumerate(ranked[: self.max_slots]):
                flat_idx = int(flat_idx_tensor.item())
                predicted[b, out_idx, : self.num_fields] = tuple_positions[b, flat_idx, : self.num_fields]
                active[b, out_idx] = True
        return predicted, active

    def _select_with_gold_fields(
        self,
        scores: torch.Tensor,
        tuple_positions: torch.Tensor,
        tuple_mask: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        gold_fields: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = scores.size(0)
        predicted = torch.full((bsz, self.max_slots, self.num_fields), -1, device=scores.device, dtype=torch.long)
        active = torch.zeros((bsz, self.max_slots), device=scores.device, dtype=torch.bool)
        for b in range(bsz):
            valid_slots = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            for out_idx, slot_tensor in enumerate(valid_slots[: self.max_slots]):
                slot = int(slot_tensor.item())
                gold = [
                    int(batch["memory_token_positions"][b, slot, 0].item()),
                    int(batch["memory_token_positions"][b, slot, 1].item()),
                ]
                if self.has_condition:
                    gold.append(int(batch["memory_condition_positions"][b, slot].item()))
                candidates = tuple_mask[b].clone()
                for field in gold_fields:
                    candidates = candidates & (tuple_positions[b, :, field] == gold[field])
                valid_indices = torch.nonzero(candidates, as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    predicted[b, out_idx, : len(gold)] = torch.tensor(gold, device=scores.device)
                else:
                    best = valid_indices[torch.argmax(scores[b, valid_indices])]
                    predicted[b, out_idx, : self.num_fields] = tuple_positions[b, best, : self.num_fields]
                for field in gold_fields:
                    predicted[b, out_idx, field] = gold[field]
                active[b, out_idx] = True
        return predicted, active

    def _score_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        scores: torch.Tensor,
        tuple_mask: torch.Tensor,
        candidate_positions: torch.Tensor,
        tuple_positions: torch.Tensor,
    ) -> Dict[str, float]:
        positive_scores = []
        negative_scores = []
        hard_fp = 0
        hard_total = 0
        labels = torch.zeros_like(scores, dtype=torch.bool)
        for b in range(scores.size(0)):
            valid_slots = torch.nonzero(batch["memory_mask"][b], as_tuple=False).flatten()
            gold_positions = []
            for slot_tensor in valid_slots:
                slot = int(slot_tensor.item())
                fields = [
                    int(batch["memory_token_positions"][b, slot, 0].item()),
                    int(batch["memory_token_positions"][b, slot, 1].item()),
                ]
                if self.has_condition:
                    fields.append(int(batch["memory_condition_positions"][b, slot].item()))
                gold_positions.append(tuple(fields))
            for idx in torch.nonzero(tuple_mask[b], as_tuple=False).flatten():
                fields = tuple(int(x) for x in tuple_positions[b, int(idx.item()), : self.num_fields].tolist())
                if fields in gold_positions:
                    labels[b, int(idx.item())] = True
        for b in range(scores.size(0)):
            pos = scores[b][tuple_mask[b] & labels[b]]
            neg = scores[b][tuple_mask[b] & ~labels[b]]
            positive_scores.extend(float(x) for x in pos.detach().cpu().tolist())
            negative_scores.extend(float(x) for x in neg.detach().cpu().tolist())
            hard_total += int(neg.numel())
            hard_fp += int((torch.sigmoid(neg) > self.tuple_threshold).sum().item())
        auc = 0.0
        if positive_scores and negative_scores:
            pair_hits = 0.0
            for pos_score in positive_scores:
                for neg_score in negative_scores:
                    pair_hits += float(pos_score > neg_score) + 0.5 * float(pos_score == neg_score)
            auc = pair_hits / (len(positive_scores) * len(negative_scores))
        pos_mean = sum(positive_scores) / max(len(positive_scores), 1)
        neg_mean = sum(negative_scores) / max(len(negative_scores), 1)
        return {
            "tuple_auc": auc,
            "tuple_positive_score_mean": pos_mean,
            "tuple_negative_score_mean": neg_mean,
            "tuple_score_margin": pos_mean - neg_mean,
            "hard_negative_false_positive_rate": hard_fp / max(hard_total, 1),
            **self._last_profile,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        candidate_mode: str,
        decode_mode: str = "top_true_count",
        threshold: float | None = None,
        tuple_pruning: str = "none",
        pair_beam_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        if decode_mode == "gold_all_fields":
            return self._gold_batch(batch)

        scores, tuple_positions, tuple_mask, candidate_positions, candidate_mask = self.score_tuples(
            batch,
            candidate_k,
            candidate_mode,
            tuple_pruning=tuple_pruning,
            pair_beam_size=pair_beam_size,
        )
        if decode_mode == "threshold":
            predicted, active = self._select_threshold(
                scores,
                tuple_positions,
                tuple_mask,
                self.tuple_threshold if threshold is None else threshold,
            )
        elif decode_mode in {"gold_key", "gold_value", "gold_cond", "gold_key_cond"}:
            field_map = {
                "gold_key": [0],
                "gold_value": [1],
                "gold_cond": [2],
                "gold_key_cond": [0, 2],
            }
            predicted, active = self._select_with_gold_fields(
                scores,
                tuple_positions,
                tuple_mask,
                batch,
                field_map[decode_mode],
            )
        else:
            predicted, active = self._select_top_true_count(scores, tuple_positions, tuple_mask, batch)
        debug_metrics = self._score_metrics(batch, scores, tuple_mask, candidate_positions, tuple_positions)
        debug_metrics.update(self._last_candidate_debug)
        return self._build_rewritten(batch, predicted, active, candidate_positions, candidate_mask, debug_metrics)

    @torch.no_grad()
    def debug_examples(
        self,
        batch: Dict[str, torch.Tensor],
        candidate_k: CandidateK,
        candidate_mode: str,
        limit: int,
    ) -> List[Dict[str, object]]:
        scores, tuple_positions, tuple_mask, candidate_positions, candidate_mask = self.score_tuples(
            batch,
            candidate_k,
            candidate_mode,
        )
        predicted, active = self._select_top_true_count(scores, tuple_positions, tuple_mask, batch)
        examples = []
        input_ids = batch["input_ids"]
        names = ["key", "value", "condition"] if self.has_condition else ["key", "value"]
        for b in range(min(batch["input_ids"].size(0), limit)):
            target_tuple_indices, source_slots = self._target_tuple_indices(batch, candidate_positions, b)
            gold_flat = set(target_tuple_indices)
            gold_fields = _gold_field_position_lists(batch, self.has_condition, b)
            gold_missing = {}
            for field, name in enumerate(names):
                candidate_set = {
                    int(pos.item())
                    for idx, pos in enumerate(candidate_positions[b, field])
                    if bool(candidate_mask[b, field, idx].item())
                }
                gold_missing[name] = [pos for pos in gold_fields[field] if pos not in candidate_set]
            true_slots = []
            for slot in source_slots:
                true_slots.append(
                    {
                        "slot": slot,
                        "key_pos": int(batch["memory_token_positions"][b, slot, 0].item()),
                        "key_token": int(input_ids[b, batch["memory_token_positions"][b, slot, 0]].item()),
                        "value_pos": int(batch["memory_token_positions"][b, slot, 1].item()),
                        "value_token": int(input_ids[b, batch["memory_token_positions"][b, slot, 1]].item()),
                        "condition_pos": int(batch["memory_condition_positions"][b, slot].item()) if self.has_condition else -1,
                        "condition_token": int(input_ids[b, batch["memory_condition_positions"][b, slot]].item()) if self.has_condition else -1,
                    }
                )
            candidates = {}
            for field, name in enumerate(names):
                candidates[name] = [
                    {"index": idx, "pos": int(pos.item()), "token": int(input_ids[b, pos].item())}
                    for idx, pos in enumerate(candidate_positions[b, field])
                    if bool(candidate_mask[b, field, idx].item())
                ]
            valid_indices = torch.nonzero(tuple_mask[b], as_tuple=False).flatten()
            ranked = valid_indices[torch.argsort(scores[b, valid_indices], descending=True)]
            top_tuples = []
            for flat_idx_tensor in ranked[:10]:
                flat_idx = int(flat_idx_tensor.item())
                fields = [int(x) for x in tuple_positions[b, flat_idx, : self.num_fields].tolist()]
                top_tuples.append(
                    {
                        "flat_index": flat_idx,
                        "positions": fields,
                        "tokens": [int(input_ids[b, pos].item()) if pos >= 0 else -1 for pos in fields],
                        "score": float(scores[b, flat_idx].item()),
                        "prob": float(torch.sigmoid(scores[b, flat_idx]).item()),
                        "is_gold": flat_idx in gold_flat,
                    }
                )
            pred_items = []
            for slot in range(self.max_slots):
                if bool(active[b, slot].item()):
                    fields = [int(x) for x in predicted[b, slot, : self.num_fields].tolist()]
                    pred_items.append(
                        {
                            "slot": slot,
                            "positions": fields,
                            "tokens": [int(input_ids[b, pos].item()) if pos >= 0 else -1 for pos in fields],
                        }
                    )
            gold_items = {
                (
                    int(input_ids[b, batch["memory_token_positions"][b, slot, 0]].item()),
                    int(input_ids[b, batch["memory_token_positions"][b, slot, 1]].item()),
                    int(input_ids[b, batch["memory_condition_positions"][b, slot]].item()) if self.has_condition else -1,
                )
                for slot in source_slots
            }
            pred_tuple_items = {(item["tokens"][0], item["tokens"][1], item["tokens"][2] if self.has_condition else -1) for item in pred_items}
            if pred_tuple_items == gold_items:
                failure_reason = "exact"
            elif gold_missing.get("key"):
                failure_reason = "missing_key_candidate"
            elif gold_missing.get("condition"):
                failure_reason = "missing_condition_candidate"
            elif gold_missing.get("value"):
                failure_reason = "missing_value_candidate"
            elif len(pred_items) != len(gold_items):
                failure_reason = "threshold/cardinality_error"
            else:
                failure_reason = "tuple_scoring_error"
            examples.append(
                {
                    "sample": b,
                    "token_sequence": input_ids[b].tolist(),
                    "true_slots": true_slots,
                    "candidates": candidates,
                    "gold_fields_missing": gold_missing,
                    "target_tuple_indices": target_tuple_indices,
                    "top_scored_tuples": top_tuples,
                    "selected_predictions": pred_items,
                    "failure_reason": failure_reason,
                }
            )
        return examples


def save_writer_v4_checkpoint(path: str, model: "ContextualTupleEdgeScorer", config: dict) -> None:
    """Save a trained Writer V4 to disk: weights + the config dict you used
    to construct it (the same kwargs you passed to ContextualTupleEdgeScorer(...),
    e.g. {"max_slots": 8, "seq_len": 512, "has_condition": True}).
    Use load_writer_v4_checkpoint to get it back."""
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_writer_v4_checkpoint(path: str, map_location: str = "cpu") -> "ContextualTupleEdgeScorer":
    """Rebuild a Writer V4 exactly as it was saved by save_writer_v4_checkpoint."""
    payload = torch.load(path, map_location=map_location)
    model = ContextualTupleEdgeScorer(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model


def _field_tokens(batch: Dict[str, torch.Tensor], positions: torch.Tensor) -> torch.Tensor:
    bsz = batch["input_ids"].size(0)
    safe = positions.clamp_min(0).clamp_max(batch["input_ids"].size(1) - 1)
    tokens = batch["input_ids"][torch.arange(bsz, device=positions.device)[:, None], safe]
    return torch.where(positions >= 0, tokens, torch.full_like(tokens, -1))


def slot_token_tuples(batch: Dict[str, torch.Tensor], has_condition: bool) -> torch.Tensor:
    keys = _field_tokens(batch, batch["memory_token_positions"][:, :, 0])
    values = _field_tokens(batch, batch["memory_token_positions"][:, :, 1])
    if has_condition:
        conditions = _field_tokens(batch, batch["memory_condition_positions"])
    else:
        conditions = torch.full_like(keys, -1)
    return torch.stack([keys, values, conditions], dim=-1)


def writer_quality_metrics(
    oracle: Dict[str, torch.Tensor],
    written: Dict[str, torch.Tensor],
    has_condition: bool,
) -> Dict[str, float]:
    gold = slot_token_tuples(oracle, has_condition)
    pred = slot_token_tuples(written, has_condition)
    gold_mask = oracle["memory_mask"]
    pred_mask = written["memory_mask"]
    bsz, gold_slots, _ = gold.shape
    pred_slots = pred.shape[1]

    key_hits = []
    value_hits = []
    condition_hits = []
    full_hits = []
    all_exact = []
    precision_hits = 0
    recall_hits = 0
    pred_total = 0
    gold_total = 0
    duplicate_count = 0
    post_query_leaks = 0
    answer_leaks = 0
    position_total = 0
    query_positions = first_positions(written["input_ids"], QUERY)
    slot_count_hits = []
    over_count = 0
    under_count = 0
    objectness_hits = []
    objectness_tp = 0
    objectness_fp = 0
    objectness_fn = 0
    true_objectness_scores = []
    false_objectness_scores = []
    objectness_probs = written.get("objectness_probs")
    objectness_targets = written.get("objectness_target_mask")
    candidate_positions = written.get("candidate_positions")
    candidate_mask = written.get("candidate_mask")
    candidate_gold_hits = [0, 0, 0]
    candidate_gold_total = [0, 0, 0]
    candidate_total = [0, 0, 0]
    candidate_false = 0
    tuple_scoring_error_count = 0
    exact_slot_available_hits = 0
    exact_slot_available_total = 0

    for b in range(bsz):
        gold_items = []
        pred_items = []
        for slot in range(gold_slots):
            if bool(gold_mask[b, slot].item()):
                gold_tuple = tuple(int(x) for x in gold[b, slot].tolist())
                gold_items.append(gold_tuple)
        for slot in range(pred_slots):
            if bool(pred_mask[b, slot].item()):
                pred_tuple = tuple(int(x) for x in pred[b, slot].tolist())
                pred_items.append(pred_tuple)

        gold_set = set(gold_items)
        pred_set = set(pred_items)
        if candidate_positions is not None and candidate_mask is not None:
            gold_fields = _gold_field_position_lists(oracle, has_condition, b)
            all_gold_fields_present = True
            candidate_field_sets: List[set[int]] = []
            for field, positions_for_field in enumerate(gold_fields):
                gold_field_set = set(positions_for_field)
                candidate_field_items = [
                    int(pos.item())
                    for pos in candidate_positions[b, field, candidate_mask[b, field]]
                ]
                candidate_field_set = set(candidate_field_items)
                candidate_field_sets.append(candidate_field_set)
                candidate_gold_hits[field] += len([pos for pos in positions_for_field if pos in candidate_field_set])
                candidate_gold_total[field] += len(positions_for_field)
                candidate_total[field] += len(candidate_field_items)
                candidate_false += len([pos for pos in candidate_field_items if pos not in gold_field_set])
                all_gold_fields_present = all_gold_fields_present and all(pos in candidate_field_set for pos in positions_for_field)
            for slot in range(gold_slots):
                if not bool(gold_mask[b, slot].item()):
                    continue
                slot_positions = [
                    int(oracle["memory_token_positions"][b, slot, 0].item()),
                    int(oracle["memory_token_positions"][b, slot, 1].item()),
                ]
                if has_condition:
                    slot_positions.append(int(oracle["memory_condition_positions"][b, slot].item()))
                exact_slot_available_total += 1
                if all(
                    pos >= 0 and field < len(candidate_field_sets) and pos in candidate_field_sets[field]
                    for field, pos in enumerate(slot_positions)
                ):
                    exact_slot_available_hits += 1
            if all_gold_fields_present and pred_set != gold_set:
                tuple_scoring_error_count += 1
        precision_hits += len([item for item in pred_items if item in gold_set])
        recall_hits += len([item for item in gold_items if item in pred_set])
        pred_total += len(pred_items)
        gold_total += len(gold_items)
        duplicate_count += max(len(pred_items) - len(pred_set), 0)
        all_exact.append(float(pred_set == gold_set))
        slot_count_hits.append(float(len(pred_items) == len(gold_items)))
        over_count += max(len(pred_items) - len(gold_items), 0)
        under_count += max(len(gold_items) - len(pred_items), 0)

        unused_pred = set(range(len(pred_items)))
        for gold_item in gold_items:
            if not pred_items:
                key_hits.append(0.0)
                value_hits.append(0.0)
                if has_condition:
                    condition_hits.append(0.0)
                full_hits.append(0.0)
                continue
            best_index = None
            best_score = -1
            for pred_index, pred_item in enumerate(pred_items):
                if pred_index not in unused_pred:
                    continue
                score = int(pred_item[0] == gold_item[0]) + int(pred_item[1] == gold_item[1])
                if has_condition:
                    score += int(pred_item[2] == gold_item[2])
                if score > best_score:
                    best_score = score
                    best_index = pred_index
            if best_index is None:
                best_index = 0
            else:
                unused_pred.remove(best_index)
            pred_item = pred_items[best_index]
            key_hits.append(float(pred_item[0] == gold_item[0]))
            value_hits.append(float(pred_item[1] == gold_item[1]))
            if has_condition:
                condition_hits.append(float(pred_item[2] == gold_item[2]))
            full_hits.append(float(pred_item == gold_item))

        for slot in range(pred_slots):
            pred_tuple = tuple(int(x) for x in pred[b, slot].tolist())
            if objectness_targets is not None:
                target_active = bool(objectness_targets[b, slot].item())
            else:
                target_active = pred_tuple in gold_set
            predicted_active = bool(pred_mask[b, slot].item())
            objectness_hits.append(float(predicted_active == target_active))
            objectness_tp += int(predicted_active and target_active)
            objectness_fp += int(predicted_active and not target_active)
            objectness_fn += int((not predicted_active) and target_active)
            if objectness_probs is not None:
                score = float(objectness_probs[b, slot].item())
                if target_active:
                    true_objectness_scores.append(score)
                else:
                    false_objectness_scores.append(score)

        answer_positions = set((oracle["answer_target_positions"][b] + 1).tolist())
        active = pred_mask[b]
        positions = written["memory_token_positions"][b, active].reshape(-1)
        if has_condition and "memory_condition_positions" in written:
            cond = written["memory_condition_positions"][b, active]
            positions = torch.cat([positions, cond[cond >= 0]])
        position_total += int(positions.numel())
        if positions.numel() > 0:
            post_query_leaks += int((positions >= query_positions[b]).sum().item())
            answer_leaks += sum(int(pos.item()) in answer_positions for pos in positions)

    slot_precision = precision_hits / max(pred_total, 1)
    slot_recall = recall_hits / max(gold_total, 1)
    slot_f1 = 0.0 if slot_precision + slot_recall == 0 else 2 * slot_precision * slot_recall / (slot_precision + slot_recall)
    false_slots = max(pred_total - precision_hits, 0)
    missed_slots = max(gold_total - recall_hits, 0)
    objectness_precision = objectness_tp / max(objectness_tp + objectness_fp, 1)
    objectness_recall = objectness_tp / max(objectness_tp + objectness_fn, 1)
    objectness_f1 = (
        0.0
        if objectness_precision + objectness_recall == 0
        else 2 * objectness_precision * objectness_recall / (objectness_precision + objectness_recall)
    )
    mean_true_objectness = sum(true_objectness_scores) / max(len(true_objectness_scores), 1)
    mean_false_objectness = sum(false_objectness_scores) / max(len(false_objectness_scores), 1)
    if true_objectness_scores and false_objectness_scores:
        pair_hits = 0.0
        for true_score in true_objectness_scores:
            for false_score in false_objectness_scores:
                pair_hits += float(true_score > false_score) + 0.5 * float(true_score == false_score)
        objectness_auc = pair_hits / (len(true_objectness_scores) * len(false_objectness_scores))
    else:
        objectness_auc = 0.0
    candidate_total_all = sum(candidate_total)
    candidate_hit_all = sum(candidate_gold_hits)
    candidate_gold_all = sum(candidate_gold_total)
    candidate_precision = candidate_hit_all / max(candidate_total_all, 1)
    candidate_false_positive_rate = candidate_false / max(candidate_total_all, 1)
    candidate_key_recall = candidate_gold_hits[0] / max(candidate_gold_total[0], 1)
    candidate_value_recall = candidate_gold_hits[1] / max(candidate_gold_total[1], 1)
    candidate_key_precision = candidate_gold_hits[0] / max(candidate_total[0], 1)
    candidate_value_precision = candidate_gold_hits[1] / max(candidate_total[1], 1)
    if has_condition:
        candidate_condition_recall = candidate_gold_hits[2] / max(candidate_gold_total[2], 1)
        candidate_condition_precision = candidate_gold_hits[2] / max(candidate_total[2], 1)
        candidate_pool_condition = candidate_total[2] / max(bsz, 1)
    else:
        candidate_condition_recall = 1.0
        candidate_condition_precision = 1.0
        candidate_pool_condition = 0.0
    all_fields_candidate_recall = candidate_hit_all / max(candidate_gold_all, 1)
    candidate_miss_rate_key = 1.0 - candidate_key_recall
    candidate_miss_rate_value = 1.0 - candidate_value_recall
    candidate_miss_rate_condition = 1.0 - candidate_condition_recall
    return {
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "key_accuracy": sum(key_hits) / max(len(key_hits), 1),
        "condition_accuracy": (sum(condition_hits) / max(len(condition_hits), 1)) if has_condition else 1.0,
        "value_accuracy": sum(value_hits) / max(len(value_hits), 1),
        "full_slot_exact": sum(full_hits) / max(len(full_hits), 1),
        "all_slots_exact": sum(all_exact) / max(len(all_exact), 1),
        "false_slot_rate": false_slots / max(pred_total, 1),
        "missed_slot_rate": missed_slots / max(gold_total, 1),
        "duplicate_slot_rate": duplicate_count / max(pred_total, 1),
        "predicted_slot_count": pred_total / max(bsz, 1),
        "true_slot_count": gold_total / max(bsz, 1),
        "slot_count_accuracy": sum(slot_count_hits) / max(len(slot_count_hits), 1),
        "overprediction_rate": over_count / max(gold_total, 1),
        "underprediction_rate": under_count / max(gold_total, 1),
        "objectness_accuracy": sum(objectness_hits) / max(len(objectness_hits), 1),
        "objectness_auc": objectness_auc,
        "objectness_precision": objectness_precision,
        "objectness_recall": objectness_recall,
        "objectness_f1": objectness_f1,
        "mean_objectness_true_slots": mean_true_objectness,
        "mean_objectness_false_slots": mean_false_objectness,
        "objectness_margin": mean_true_objectness - mean_false_objectness,
        "candidate_key_recall": candidate_key_recall,
        "candidate_condition_recall": candidate_condition_recall,
        "candidate_value_recall": candidate_value_recall,
        "all_fields_candidate_recall": all_fields_candidate_recall,
        "learned_candidate_key_recall": candidate_key_recall,
        "learned_candidate_cond_recall": candidate_condition_recall,
        "learned_candidate_value_recall": candidate_value_recall,
        "learned_candidate_all_field_recall": all_fields_candidate_recall,
        "learned_candidate_key_precision": candidate_key_precision,
        "learned_candidate_cond_precision": candidate_condition_precision,
        "learned_candidate_value_precision": candidate_value_precision,
        "candidate_pool_size_key": candidate_total[0] / max(bsz, 1),
        "candidate_pool_size_condition": candidate_pool_condition,
        "candidate_pool_size_value": candidate_total[1] / max(bsz, 1),
        "candidate_pool_size_cond": candidate_pool_condition,
        "exact_slot_available_rate": exact_slot_available_hits / max(exact_slot_available_total, 1),
        "candidate_precision": candidate_precision,
        "candidate_false_positive_rate": candidate_false_positive_rate,
        "candidate_miss_rate": 1.0 - all_fields_candidate_recall,
        "candidate_miss_rate_key": candidate_miss_rate_key,
        "candidate_miss_rate_cond": candidate_miss_rate_condition,
        "candidate_miss_rate_value": candidate_miss_rate_value,
        "candidate_miss_rate_any": 1.0 - all_fields_candidate_recall,
        "condition_miss_rate": candidate_miss_rate_condition,
        "value_miss_rate": candidate_miss_rate_value,
        "tuple_scoring_error_rate": tuple_scoring_error_count / max(bsz, 1),
        "post_query_leak_rate": post_query_leaks / max(position_total, 1),
        "answer_token_leak_rate": answer_leaks / max(position_total, 1),
    }
