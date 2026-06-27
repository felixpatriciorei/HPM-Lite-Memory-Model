from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .data import VOCAB_SIZE
from .memory import EpisodicMemory, HebbianMemory


def make_local_causal_mask(seq_len: int, window: int, device: torch.device | None = None) -> torch.Tensor:
    """Return [T, T] mask where row t can attend only max(0, t-W)..t."""

    idx = torch.arange(seq_len, device=device)
    query = idx[:, None]
    key = idx[None, :]
    return (key <= query) & ((query - key) <= window)


class LocalCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # True sliding-window causal attention.
        #
        # The previous implementation formed a dense [B, H, T, T] score
        # matrix and then masked out non-local positions. That is logically
        # local attention, but its memory cost is still quadratic in T. At
        # T=4096, B=32, H=4, the score tensor alone is about 8 GiB.
        #
        # This chunked gather computes only the last ``window`` keys for each
        # query, so attention memory is O(B * H * T * window), not O(B*H*T*T).
        radius = min(max(int(self.window), 0), max(seq_len - 1, 0))
        chunk_size = 512
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)

        for start in range(0, seq_len, chunk_size):
            end = min(seq_len, start + chunk_size)
            chunk_len = end - start
            positions = torch.arange(start, end, device=x.device)
            offsets = torch.arange(radius, -1, -1, device=x.device)
            key_positions = positions[:, None] - offsets[None, :]
            valid = key_positions >= 0
            gather_positions = key_positions.clamp_min(0)

            gather_index = gather_positions[None, None, :, :, None].expand(
                bsz, self.heads, chunk_len, radius + 1, self.head_dim
            )
            k_chunk = torch.gather(
                k.unsqueeze(2).expand(bsz, self.heads, chunk_len, seq_len, self.head_dim),
                dim=3,
                index=gather_index,
            )
            v_chunk = torch.gather(
                v.unsqueeze(2).expand(bsz, self.heads, chunk_len, seq_len, self.head_dim),
                dim=3,
                index=gather_index,
            )

            q_chunk = q[:, :, start:end, :]
            scores = torch.sum(q_chunk.unsqueeze(-2) * k_chunk, dim=-1) * scale
            scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            outputs.append(torch.sum(attn.unsqueeze(-1) * v_chunk, dim=-2))

        y = torch.cat(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, window: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LocalCausalSelfAttention(d_model, heads, window, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleRecurrentSummary(nn.Module):
    """EMA summary over completed blocks only, kept for the old recurrent baseline."""

    def __init__(self, d_model: int, block_size: int, decay: float = 0.9):
        super().__init__()
        self.block_size = block_size
        self.decay = decay
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = hidden.shape
        state = hidden.new_zeros(bsz, d_model)
        summaries = hidden.new_zeros(hidden.shape)
        for start in range(0, seq_len, self.block_size):
            end = min(seq_len, start + self.block_size)
            summaries[:, start:end, :] = state[:, None, :]
            block_summary = hidden[:, start:end, :].mean(dim=1)
            state = self.decay * state + (1.0 - self.decay) * block_summary
        return self.proj(summaries)


class GruRecurrentState(nn.Module):
    """Causal recurrent memory path: x_t -> GRU state r_t."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.ln(x))
        return self.out(self.dropout(states))


class MemoryPathRouter(nn.Module):
    """alpha = softmax(W[l_t, r_t, e_t]); m_t = sum_i alpha_i path_i."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(3 * d_model, 3)

    def forward(
        self,
        local_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        episodic_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(torch.cat([local_state, recurrent_state, episodic_state], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        mixed = (
            weights[..., 0:1] * local_state
            + weights[..., 1:2] * recurrent_state
            + weights[..., 2:3] * episodic_state
        )
        return mixed, weights


@dataclass
class HpmLiteConfig:
    model_type: str = "local"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    window: int = 64
    max_seq_len: int = 2048
    dropout: float = 0.0
    hebbian_decay: float = 0.9
    hebbian_eta: float = 1.0
    use_null_slot: bool = False
    null_score_init: float = 0.0


@dataclass
class AnswerTransformerConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    layers: int = 1
    heads: int = 4
    window: int = 64
    max_seq_len: int = 2048
    dropout: float = 0.0


class AnswerTransformerModel(nn.Module):
    """Small causal local-window Transformer used as a no-memory answer baseline."""

    def __init__(self, config: AnswerTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.answer_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.answer_head(self.final_ln(x))


class HpmLiteModel(nn.Module):
    def __init__(self, config: HpmLiteConfig):
        super().__init__()
        if config.model_type not in {"local", "recurrent", "epmem", "hpm_lite", "hebbian"}:
            raise ValueError(f"unknown model_type: {config.model_type}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )

        # Legacy baseline: local mixer + simple recurrent summary.
        self.recurrent = (
            SimpleRecurrentSummary(config.d_model, block_size=config.window)
            if config.model_type == "recurrent"
            else None
        )

        # The actual screenshot model: local mixer, GRU state, episodic retrieval, learned router.
        self.hpm_gru = GruRecurrentState(config.d_model, config.dropout) if config.model_type == "hpm_lite" else None
        self.router = MemoryPathRouter(config.d_model) if config.model_type == "hpm_lite" else None

        if config.model_type in {"epmem", "hpm_lite"}:
            self.memory = EpisodicMemory(
                config.d_model,
                use_null_slot=config.use_null_slot,
                null_score_init=config.null_score_init,
            )
        elif config.model_type == "hebbian":
            self.memory = HebbianMemory(config.d_model, decay=config.hebbian_decay, eta=config.hebbian_eta)
        else:
            self.memory = None

        # Kept for old diagnostic variants; hpm_lite uses router weights instead.
        self.gamma_e = nn.Parameter(torch.tensor(1.0))
        self.gamma_r = nn.Parameter(torch.tensor(0.5))
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _retrieve_answer_memory(
        self,
        token_emb: torch.Tensor,
        memory_token_positions: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
        query_key_positions: Optional[torch.Tensor],
        top_k: int,
        task: str,
        hop_positive_memory_indices: Optional[torch.Tensor],
        memory_control: str,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.memory is None:
            raise RuntimeError("_retrieve_answer_memory called without a memory module")
        if memory_token_positions is None or memory_mask is None or query_key_positions is None:
            raise ValueError("memory models require memory positions and query key positions")
        num_hops = 2 if task in {"twohop", "longhop"} else 1
        return self.memory(
            token_embeddings=token_emb,
            memory_token_positions=memory_token_positions,
            memory_mask=memory_mask,
            query_key_positions=query_key_positions,
            top_k=top_k,
            num_hops=num_hops,
            hop_positive_indices=hop_positive_memory_indices,
            memory_control=memory_control,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_token_positions: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        answer_positions: Optional[torch.Tensor] = None,
        query_key_positions: Optional[torch.Tensor] = None,
        top_k: int = 1,
        task: str = "kv",
        hop_positive_memory_indices: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device)
        token_emb = self.token_embedding(input_ids)
        embedded = token_emb + self.position_embedding(positions)[None, :, :]

        local_state = embedded
        for block in self.blocks:
            local_state = block(local_state)

        retrieval_info: Dict[str, torch.Tensor] = {}
        state = local_state

        if self.config.model_type == "recurrent" and self.recurrent is not None:
            state = local_state + self.gamma_r * self.recurrent(local_state)

        elif self.config.model_type == "hpm_lite":
            if self.hpm_gru is None or self.router is None or self.memory is None:
                raise RuntimeError("hpm_lite model was not initialized correctly")
            if answer_positions is None:
                raise ValueError("hpm_lite requires answer_positions so episodic retrieval can be routed at the answer token")

            recurrent_state = self.hpm_gru(embedded)
            episodic_state = torch.zeros_like(local_state)
            retrieved, retrieval_info = self._retrieve_answer_memory(
                token_emb=token_emb,
                memory_token_positions=memory_token_positions,
                memory_mask=memory_mask,
                query_key_positions=query_key_positions,
                top_k=top_k,
                task=task,
                hop_positive_memory_indices=hop_positive_memory_indices,
                memory_control=memory_control,
            )
            batch = torch.arange(bsz, device=input_ids.device)
            episodic_state[batch, answer_positions] = retrieved
            state, router_weights = self.router(local_state, recurrent_state, episodic_state)
            retrieval_info = dict(retrieval_info)
            retrieval_info["router_weights"] = router_weights

        elif self.memory is not None:
            if answer_positions is None:
                raise ValueError("memory models require answer_positions")
            retrieved, retrieval_info = self._retrieve_answer_memory(
                token_emb=token_emb,
                memory_token_positions=memory_token_positions,
                memory_mask=memory_mask,
                query_key_positions=query_key_positions,
                top_k=top_k,
                task=task,
                hop_positive_memory_indices=hop_positive_memory_indices,
                memory_control=memory_control,
            )
            batch = torch.arange(bsz, device=input_ids.device)
            state = local_state.clone()
            state[batch, answer_positions] = state[batch, answer_positions] + self.gamma_e * retrieved

        logits = self.lm_head(self.final_ln(state))
        return {"logits": logits, "retrieval": retrieval_info}
