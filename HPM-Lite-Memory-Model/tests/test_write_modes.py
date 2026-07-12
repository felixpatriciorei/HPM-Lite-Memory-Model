import torch

from hpm_lite.data import ANSWER, QUERY, FactRecallConfig, FactRecallDataset
from hpm_lite.write_modes import apply_write_mode, first_positions


def test_fact_token_writer_matches_clean_oracle_without_metadata_writes():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=21, task="kv"))
    batch = dataset.sample_batch(4)
    written, stats = apply_write_mode(batch, "fact_token")

    assert torch.equal(written["memory_token_positions"], batch["memory_token_positions"])
    assert torch.equal(written["memory_mask"], batch["memory_mask"])
    assert stats["true_fact_written_rate"] == 1.0
    assert stats["false_write_rate"] == 0.0
    assert stats["missed_fact_rate"] == 0.0


def test_fact_token_writer_preserves_multi_positive_mask():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=224, window=64, seed=23, task="coexisting"))
    batch = dataset.sample_batch(4)
    written, stats = apply_write_mode(batch, "fact_token")

    assert torch.equal(written["positive_memory_mask"], batch["positive_memory_mask"])
    assert torch.all(written["positive_memory_mask"].sum(dim=1) == 2)
    assert stats["true_fact_written_rate"] == 1.0


def test_random_write_never_uses_query_answer_or_post_query_tokens():
    dataset = FactRecallDataset(FactRecallConfig(seq_len=192, window=64, seed=22, task="kv"))
    batch = dataset.sample_batch(8)
    written, _ = apply_write_mode(batch, "random_write")
    query_positions = first_positions(written["input_ids"], QUERY)

    for b in range(written["input_ids"].size(0)):
        valid = written["memory_mask"][b]
        positions = written["memory_token_positions"][b, valid]
        assert torch.all(positions < query_positions[b])
        stored_tokens = written["input_ids"][b, positions.reshape(-1)]
        assert QUERY not in stored_tokens.tolist()
        assert ANSWER not in stored_tokens.tolist()
