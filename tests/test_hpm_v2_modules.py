import torch

from hpm_lite.hpm_v2 import (
    BlockwiseSelectiveRecurrentState,
    FastWeightBlockMemory,
    HpmV2PathRouter,
    JepaLiteAuxiliary,
    block_summaries,
)
from hpm_lite.llm_memory_adapter import HpmTextMemory, build_memory_augmented_prompt


def test_blockwise_selective_state_shape_and_causality():
    torch.manual_seed(0)
    module = BlockwiseSelectiveRecurrentState(d_model=16, block_size=4)
    x = torch.randn(2, 12, 16)
    y = module(x)
    assert y.shape == x.shape
    # First block sees zero prior recurrent state before output projection.
    # The projection bias can make it nonzero, but all tokens in the first block
    # receive the same prior state and therefore same recurrent output.
    assert torch.allclose(y[:, 0, :], y[:, 3, :], atol=1e-5)
    assert not torch.allclose(y[:, 0, :], y[:, 4, :])


def test_fast_weight_block_memory_shape_and_gradients():
    torch.manual_seed(1)
    module = FastWeightBlockMemory(d_model=12, block_size=3)
    x = torch.randn(2, 9, 12, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_router_mixes_four_paths():
    router = HpmV2PathRouter(d_model=8, num_paths=4)
    paths = [torch.randn(2, 5, 8) for _ in range(4)]
    mixed, weights = router(*paths)
    assert mixed.shape == paths[0].shape
    assert weights.shape == (2, 5, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_jepa_lite_auxiliary_is_finite():
    torch.manual_seed(2)
    x = torch.randn(2, 16, 20)
    summaries = block_summaries(x, block_size=4)
    jepa = JepaLiteAuxiliary(d_model=20, latent_dim=10)
    info = jepa(summaries)
    assert set(info) == {"jepa_loss", "jepa_cosine", "jepa_target_std"}
    assert torch.isfinite(info["jepa_loss"])
    assert torch.isfinite(info["jepa_cosine"])
    assert info["jepa_target_std"] > 0


def test_text_memory_adapter_fact_retrieval():
    memory = HpmTextMemory()
    assert memory.ingest_fact_syntax("FACT k12 v77\nNOISE\nFACT k03 v19") == 2
    assert memory.answer_fact_query("QUERY k03") == "v19"
    prompt = build_memory_augmented_prompt("What is k03?", memory.retrieve("k03", top_k=2))
    assert "k03" in prompt and "v19" in prompt
