from __future__ import annotations

import argparse
import json

import torch

from hpm_lite.hpm_v2 import (
    BlockwiseSelectiveRecurrentState,
    FastWeightBlockMemory,
    HpmV2PathRouter,
    JepaLiteAuxiliary,
    block_summaries,
)
from hpm_lite.llm_memory_adapter import HpmTextMemory, build_memory_augmented_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test HPM-Lite v2 memory modules.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)
    selective = BlockwiseSelectiveRecurrentState(args.d_model, args.block_size).to(device)
    fast = FastWeightBlockMemory(args.d_model, args.block_size).to(device)
    router = HpmV2PathRouter(args.d_model, num_paths=4).to(device)
    jepa = JepaLiteAuxiliary(args.d_model).to(device)

    r = selective(x)
    f = fast(x)
    e = torch.zeros_like(x)
    mixed, weights = router(x, r, f, e)
    summaries = block_summaries(mixed, args.block_size)
    jepa_info = jepa(summaries)
    loss = mixed.square().mean() + 0.01 * jepa_info["jepa_loss"]
    loss.backward()

    memory = HpmTextMemory()
    memory.ingest_fact_syntax("FACT project_code falcon17\nFACT owner felix")
    prompt = build_memory_augmented_prompt("What is project_code?", memory.retrieve("project_code"))

    print(json.dumps({
        "device": str(device),
        "input_shape": list(x.shape),
        "mixed_shape": list(mixed.shape),
        "router_weight_shape": list(weights.shape),
        "router_weight_sum_mean": float(weights.sum(dim=-1).mean().detach().cpu()),
        "jepa_loss": float(jepa_info["jepa_loss"].detach().cpu()),
        "text_memory_prompt_contains_fact": "falcon17" in prompt,
    }, indent=2))


if __name__ == "__main__":
    main()
