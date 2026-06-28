from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .data import VOCAB_SIZE
from .memory import EpisodicMemory
from .model import (
    SupervisedMemoryWriter,
    TransformerBlock,
    match_hop_positive_indices,
)
from .hpm_v2 import (
    BlockwiseSelectiveRecurrentState,
    FastWeightBlockMemory,
    HpmV2PathRouter,
    JepaLiteAuxiliary,
    block_summaries,
)


@dataclass
class HpmLiteV2Config:
    """Trainable HPM-Lite v2 model config.

    v2 keeps the v1 public forward interface so it can plug into the existing
    synthetic KV train/eval pipeline, but replaces the 3-path HPM core with a
    4-path blockwise memory core:

    local attention + selective recurrent state + fast-weight memory + episodic memory.
    """

    model_type: str = "hpm_lite_v2"
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    window: int = 256
    max_seq_len: int = 2048
    dropout: float = 0.0
    block_size: int = 128
    fast_decay_init: float = 0.95
    use_null_slot: bool = False
    null_score_init: float = 0.0
    use_learned_writer: bool = False
    use_jepa_aux: bool = True
    jepa_latent_dim: Optional[int] = None


class HpmLiteV2Model(nn.Module):
    """HPM-Lite v2, wired to the same task API as the v1 model.

    This is intentionally not a from-scratch LLM. It is the next trainable HPM
    research model for the controlled long-range memory benchmark.
    """

    def __init__(self, config: HpmLiteV2Config):
        super().__init__()
        if config.model_type != "hpm_lite_v2":
            raise ValueError(f"HpmLiteV2Model requires model_type='hpm_lite_v2', got {config.model_type!r}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.local_blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.window, config.dropout) for _ in range(config.layers)]
        )
        self.selective_recurrent = BlockwiseSelectiveRecurrentState(
            config.d_model, block_size=config.block_size, dropout=config.dropout
        )
        self.fast_memory = FastWeightBlockMemory(
            config.d_model, block_size=config.block_size, decay_init=config.fast_decay_init
        )
        self.episodic_memory = EpisodicMemory(
            config.d_model, use_null_slot=config.use_null_slot, null_score_init=config.null_score_init
        )
        self.router = HpmV2PathRouter(config.d_model, num_paths=4)
        self.writer = SupervisedMemoryWriter(config.d_model) if config.use_learned_writer else None
        self.jepa = JepaLiteAuxiliary(config.d_model, latent_dim=config.jepa_latent_dim) if config.use_jepa_aux else None

        self.final_ln = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.answer_head.weight = self.token_embedding.weight

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

    def _local_path(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._embed(input_ids)
        for block in self.local_blocks:
            hidden = block(hidden)
        return hidden

    def _maybe_select_writer_memory(
        self,
        *,
        local_state: torch.Tensor,
        input_ids: torch.Tensor,
        query_key_positions: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        hop_positive_memory_indices: Optional[torch.Tensor],
        use_learned_writer: bool,
        learned_writer_teacher_forcing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        info: Dict[str, torch.Tensor] = {}
        active_positions = memory_token_positions
        active_mask = memory_mask
        active_positive = hop_positive_memory_indices

        if use_learned_writer and self.writer is not None:
            writer_info = self.writer(
                local_state,
                input_ids,
                query_key_positions,
                memory_token_positions,
                memory_mask,
                max_slots=memory_token_positions.size(1),
            )
            info.update(writer_info)
            if not learned_writer_teacher_forcing:
                active_positions = writer_info["writer_memory_token_positions"]
                active_mask = writer_info["writer_memory_mask"]
                active_positive = match_hop_positive_indices(
                    active_positions,
                    active_mask,
                    memory_token_positions,
                    hop_positive_memory_indices,
                )

        return active_positions, active_mask, active_positive, info

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_token_positions: torch.Tensor,
        memory_mask: torch.Tensor,
        answer_positions: torch.Tensor,
        query_key_positions: torch.Tensor,
        top_k: int = 1,
        task: str = "kv",
        hop_positive_memory_indices: Optional[torch.Tensor] = None,
        memory_control: str = "normal",
        use_learned_writer: bool = False,
        learned_writer_teacher_forcing: bool = False,
    ) -> Dict[str, Any]:
        del answer_positions  # The training loop computes loss from logits/targets.

        local_state = self._local_path(input_ids)
        recurrent_state = self.selective_recurrent(local_state)
        fast_state = self.fast_memory(local_state)

        active_positions, active_mask, active_positive, retrieval_info = self._maybe_select_writer_memory(
            local_state=local_state,
            input_ids=input_ids,
            query_key_positions=query_key_positions,
            memory_token_positions=memory_token_positions,
            memory_mask=memory_mask,
            hop_positive_memory_indices=hop_positive_memory_indices,
            use_learned_writer=use_learned_writer,
            learned_writer_teacher_forcing=learned_writer_teacher_forcing,
        )

        num_hops = 2 if task in {"twohop", "longhop"} else 1
        episodic_vector, episodic_info = self.episodic_memory(
            local_state,
            active_positions,
            active_mask,
            query_key_positions,
            top_k=top_k,
            num_hops=num_hops,
            hop_positive_indices=active_positive,
            memory_control=memory_control,
        )
        retrieval_info.update(episodic_info)

        episodic_state = episodic_vector[:, None, :].expand_as(local_state)
        mixed, router_weights = self.router(local_state, recurrent_state, fast_state, episodic_state)
        retrieval_info["router_weights"] = router_weights

        if self.jepa is not None:
            jepa_info = self.jepa(block_summaries(local_state, self.config.block_size))
            retrieval_info.update(jepa_info)

        logits = self.answer_head(self.final_ln(mixed))
        return {"logits": logits, "retrieval": retrieval_info}
