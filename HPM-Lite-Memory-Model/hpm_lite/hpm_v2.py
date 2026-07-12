from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class BlockMemoryConfig:
    """Configuration for HPM-Lite v2 memory modules.

    These modules are intentionally small and kernel-free so they run on a single
    RTX 4060 / Kaggle T4 before any custom CUDA/Mamba kernels exist.
    """

    d_model: int = 128
    block_size: int = 128
    dropout: float = 0.0
    fast_decay_init: float = 0.95


class BlockwiseSelectiveRecurrentState(nn.Module):
    """Mamba/RWKV-inspired blockwise recurrent path.

    This is not full Mamba. It is the next correct engineering step: an
    input-conditioned state update instead of a plain GRU-only path.

    For each completed block b:
        u_b = mean hidden state for the block
        gate_b = sigmoid(W_g u_b)
        cand_b = tanh(W_c u_b)
        state_b = gate_b * state_{b-1} + (1 - gate_b) * cand_b

    Tokens inside a block can only see the state from earlier completed blocks,
    so the module remains causal at block granularity.
    """

    def __init__(self, d_model: int, block_size: int = 128, dropout: float = 0.0):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.d_model = int(d_model)
        self.block_size = int(block_size)
        self.ln = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.candidate = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, d_model]")
        bsz, seq_len, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        hidden = self.ln(hidden)
        state = hidden.new_zeros(bsz, d_model)
        states = hidden.new_zeros(bsz, seq_len, d_model)

        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            states[:, start:end, :] = state[:, None, :]
            summary = hidden[:, start:end, :].mean(dim=1)
            gate = torch.sigmoid(self.gate(summary))
            candidate = torch.tanh(self.candidate(summary))
            state = gate * state + (1.0 - gate) * candidate

        return self.out(states)


class FastWeightBlockMemory(nn.Module):
    """Small differentiable fast-weight memory path.

    This implements a kernel-free associative memory:
        M <- decay * M + eta * outer(key, value)
        read_t = q_t @ M

    The update happens once per completed block. Current block tokens read the
    memory state produced by earlier blocks only.
    """

    def __init__(self, d_model: int, block_size: int = 128, decay_init: float = 0.95):
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.d_model = int(d_model)
        self.block_size = int(block_size)
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.eta = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        # Store logit so the learned value stays in (0, 1).
        decay_init = min(max(float(decay_init), 1e-4), 0.9999)
        self.decay_logit = nn.Parameter(torch.logit(torch.tensor(decay_init)))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, seq_len, d_model]")
        bsz, seq_len, d_model = hidden.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")

        h = self.ln(hidden)
        memory = h.new_zeros(bsz, d_model, d_model)
        reads = h.new_zeros(bsz, seq_len, d_model)
        decay = torch.sigmoid(self.decay_logit)

        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            q = F.normalize(self.q_proj(h[:, start:end, :]), dim=-1)
            reads[:, start:end, :] = torch.bmm(q, memory)

            summary = h[:, start:end, :].mean(dim=1)
            key = F.normalize(self.k_proj(summary), dim=-1)
            value = self.v_proj(summary)
            eta = self.eta(summary).view(bsz, 1, 1)
            update = torch.bmm(key.unsqueeze(2), value.unsqueeze(1))
            memory = decay * memory + eta * update

        return self.out(reads)


class HpmV2PathRouter(nn.Module):
    """Route among local, selective recurrent, fast-weight, and episodic paths."""

    def __init__(self, d_model: int, num_paths: int = 4):
        super().__init__()
        self.num_paths = int(num_paths)
        self.proj = nn.Linear(num_paths * d_model, num_paths)

    def forward(self, *paths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(paths) != self.num_paths:
            raise ValueError(f"expected {self.num_paths} paths, got {len(paths)}")
        logits = self.proj(torch.cat(list(paths), dim=-1))
        weights = torch.softmax(logits, dim=-1)
        mixed = sum(weights[..., i : i + 1] * path for i, path in enumerate(paths))
        return mixed, weights


class JepaLiteAuxiliary(nn.Module):
    """JEPA-lite auxiliary objective for chunk representations.

    This is deliberately auxiliary. It should never replace the exact episodic
    memory path for key-value facts. It predicts the latent representation of a
    future block from a context block using stop-gradient targets.
    """

    def __init__(self, d_model: int, latent_dim: Optional[int] = None):
        super().__init__()
        latent_dim = int(latent_dim or d_model)
        self.context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim), nn.GELU())
        self.predictor = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))
        self.target = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, block_summaries: torch.Tensor) -> Dict[str, torch.Tensor]:
        if block_summaries.ndim != 3:
            raise ValueError("block_summaries must have shape [batch, blocks, d_model]")
        if block_summaries.size(1) < 2:
            zero = block_summaries.new_zeros(())
            return {"jepa_loss": zero, "jepa_cosine": zero, "jepa_target_std": zero}

        ctx = block_summaries[:, :-1, :]
        tgt = block_summaries[:, 1:, :]
        pred = self.predictor(self.context(ctx))
        target = self.target(tgt).detach()
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()
        cosine = (pred * target).sum(dim=-1).mean()
        target_std = target.std(dim=(0, 1)).mean()
        return {"jepa_loss": loss, "jepa_cosine": cosine, "jepa_target_std": target_std}


def block_summaries(hidden: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool hidden states into block summaries [B, num_blocks, D]."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    chunks = []
    for start in range(0, hidden.size(1), block_size):
        end = min(hidden.size(1), start + block_size)
        chunks.append(hidden[:, start:end, :].mean(dim=1))
    return torch.stack(chunks, dim=1)
