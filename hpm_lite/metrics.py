from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def answer_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    losses = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none")
    mask = loss_mask.reshape(-1)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def answer_exact_accuracy(
    logits: torch.Tensor,
    answer_positions: torch.Tensor,
    answer_tokens: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(logits.size(0), device=logits.device)
    predictions = logits[batch, answer_positions].argmax(dim=-1)
    return (predictions == answer_tokens).float().mean()


def answer_span_correct_mask(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample exact match over all masked answer target positions."""

    predictions = logits.argmax(dim=-1)
    answer_mask = loss_mask > 0
    token_correct = (predictions == target_ids) | ~answer_mask
    has_answer = answer_mask.any(dim=1)
    return token_correct.all(dim=1) & has_answer


def answer_span_exact_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return answer_span_correct_mask(logits, target_ids, loss_mask).float().mean()


def answer_predictions(logits: torch.Tensor, answer_positions: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(logits.size(0), device=logits.device)
    return logits[batch, answer_positions].argmax(dim=-1)


def retrieval_correct_mask(
    retrieval: Dict[str, torch.Tensor],
    positive_indices: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not retrieval or "top_indices" not in retrieval:
        return None
    top_indices = retrieval["top_indices"]
    if positive_mask is not None:
        valid = positive_mask.any(dim=1)
        selected = torch.zeros_like(positive_mask, dtype=torch.bool)
        selected.scatter_(1, top_indices.clamp_min(0), True)
        correct = valid & ((selected & positive_mask).sum(dim=1) == positive_mask.sum(dim=1))
        return correct
    if positive_indices is None:
        return None
    valid = positive_indices >= 0
    correct = torch.zeros_like(valid, dtype=torch.bool)
    correct[valid] = (top_indices[valid] == positive_indices[valid, None]).any(dim=1)
    return correct


def retrieval_metrics(
    retrieval: Dict[str, torch.Tensor],
    positive_indices: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    if not retrieval or "top_indices" not in retrieval:
        return {}
    top_indices = retrieval["top_indices"]
    if positive_mask is not None:
        valid = positive_mask.any(dim=1)
        if not valid.any():
            return {}
        selected = torch.zeros_like(positive_mask, dtype=torch.bool)
        selected.scatter_(1, top_indices.clamp_min(0), True)
        top1 = positive_mask.gather(1, top_indices[:, :1]).squeeze(1)
        contains_any = (selected & positive_mask).any(dim=1)
        contains_all = (selected & positive_mask).sum(dim=1) == positive_mask.sum(dim=1)
        out = {
            "retrieval_top1": top1[valid].float().mean().item(),
            "retrieval_topk": contains_all[valid].float().mean().item(),
            "retrieval_topk_any": contains_any[valid].float().mean().item(),
        }
        if "scores" in retrieval:
            scores = retrieval["scores"][valid]
            pos_mask = positive_mask[valid]
            positive_scores = scores.masked_fill(~pos_mask, 1.0e9).min(dim=-1).values
            negative_scores = scores.masked_fill(pos_mask, -1.0e9).max(dim=-1).values
            out["retrieval_margin"] = (positive_scores - negative_scores).mean().item()
        return out

    if positive_indices is None:
        return {}
    valid = positive_indices >= 0
    if not valid.any():
        return {}

    top1 = (top_indices[valid, 0] == positive_indices[valid]).float().mean().item()
    contains = (top_indices[valid] == positive_indices[valid, None]).any(dim=1).float().mean().item()
    out = {
        "retrieval_top1": top1,
        "retrieval_topk": contains,
    }
    if "scores" in retrieval:
        scores = retrieval["scores"][valid]
        pos = positive_indices[valid]
        row = torch.arange(scores.size(0), device=scores.device)
        positive_scores = scores[row, pos]
        negative_scores = scores.clone()
        negative_scores[row, pos] = -1.0e9
        margin = positive_scores - negative_scores.max(dim=-1).values
        out["retrieval_margin"] = margin.mean().item()
    return out


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
