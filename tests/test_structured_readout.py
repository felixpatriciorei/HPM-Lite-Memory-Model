import torch

from hpm_lite.data import ANSWER, QUERY, FactRecallConfig, FactRecallDataset
from hpm_lite.structured_readout import (
    LearnedConditionReader,
    LearnedSetReader,
    learned_condition_stress_metrics,
    learned_set_stress_metrics,
    symbolic_set_metrics,
    symbolic_condition_binding_metrics,
    symbolic_condition_binding_predictions,
)


def test_symbolic_condition_binding_solves_clean_conditional_slots():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=224,
            window=64,
            seed=51,
            task="conditional_contrastive",
            num_facts=4,
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(8)

    metrics = symbolic_condition_binding_metrics(batch, memory_control="normal")

    assert metrics["symbolic_readout_available"] == 1.0
    assert metrics["condition_symbolic_exact"] == 1.0
    assert metrics["condition_symbolic_slot_accuracy"] == 1.0
    assert metrics["condition_symbolic_value_accuracy"] == 1.0
    assert metrics["exact_match_available_rate"] == 1.0
    assert metrics["ambiguous_exact_match_rate"] == 0.0
    assert metrics["symbolic_binding_hit_1_rate"] == 1.0


def test_symbolic_condition_binding_respects_memory_controls():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=224,
            window=64,
            seed=52,
            task="conditional_positive_only",
            num_facts=4,
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(8)

    shuffled = symbolic_condition_binding_metrics(batch, memory_control="shuffled_values")
    random_keys = symbolic_condition_binding_metrics(batch, memory_control="random_keys")
    no_retrieval = symbolic_condition_binding_metrics(batch, memory_control="no_retrieval")

    assert shuffled["exact_match_available_rate"] == 1.0
    assert shuffled["condition_symbolic_slot_accuracy"] == 1.0
    assert shuffled["condition_symbolic_value_accuracy"] == 0.0
    assert shuffled["condition_symbolic_exact"] == 0.0

    assert random_keys["exact_match_available_rate"] == 0.0
    assert random_keys["condition_symbolic_slot_accuracy"] == 0.0
    assert random_keys["condition_symbolic_exact"] == 0.0

    assert no_retrieval == {"symbolic_readout_available": 0.0}


def test_symbolic_condition_binding_uses_only_pre_query_memory():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=224,
            window=64,
            seed=53,
            task="conditional_contrastive",
            num_facts=4,
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(8)
    predictions = symbolic_condition_binding_predictions(batch, memory_control="normal")

    for b in range(batch["input_ids"].size(0)):
        answer_pos = batch["answer_positions"][b].item()
        query_pos = answer_pos - 3
        assert batch["input_ids"][b, query_pos].item() == QUERY
        assert batch["input_ids"][b, answer_pos].item() == ANSWER

        active = batch["memory_mask"][b]
        spans = batch["memory_spans"][b, active]
        positions = batch["memory_token_positions"][b, active]
        assert torch.all(spans[:, 1] < query_pos)
        assert torch.all(positions < query_pos)

        answer_token_pos = answer_pos + 1
        assert not torch.any(positions == answer_pos)
        assert not torch.any(positions == answer_token_pos)
        assert predictions["predicted_values"][b].item() == batch["answer_tokens"][b].item()


def test_learned_condition_reader_loss_and_metrics_are_finite():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=224,
            window=64,
            seed=54,
            task="conditional_contrastive",
            num_facts=4,
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)
    reader = LearnedConditionReader(reader_dim=16, hidden=32, train_embeddings=False)

    loss = reader.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in reader.parameters() if parameter.requires_grad)

    metrics = reader.metrics(batch)
    assert metrics["learned_readout_available"] == 1.0
    assert 0.0 <= metrics["learned_condition_exact"] <= 1.0
    assert 0.0 <= metrics["learned_condition_slot_accuracy"] <= 1.0
    assert 0.0 <= metrics["learned_condition_value_accuracy"] <= 1.0
    assert reader.metrics(batch, memory_control="no_retrieval") == {"learned_readout_available": 0.0}


def test_learned_set_reader_loss_and_symbolic_set_upper_bound():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=224,
            window=64,
            seed=55,
            task="coexisting",
            num_facts=4,
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)
    reader = LearnedSetReader(reader_dim=16, hidden=32, train_embeddings=False)

    symbolic = symbolic_set_metrics(batch)
    assert symbolic["symbolic_readout_available"] == 1.0
    assert symbolic["symbolic_set_exact"] == 1.0
    assert symbolic["symbolic_set_f1"] == 1.0

    loss = reader.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in reader.parameters() if parameter.requires_grad)

    metrics = reader.metrics(batch)
    assert metrics["learned_readout_available"] == 1.0
    assert 0.0 <= metrics["learned_set_exact"] <= 1.0
    assert 0.0 <= metrics["learned_set_f1"] <= 1.0
    assert reader.metrics(batch, memory_control="no_retrieval") == {"learned_readout_available": 0.0}


def test_stress_condition_symbolic_controls_and_metrics():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=81,
            task="conditional_contrastive_stress",
            num_facts=8,
            num_positive=1,
            num_hard_negatives=4,
            similarity_mode="adjacent",
            slot_order="random",
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)
    reader = LearnedConditionReader(reader_dim=16, hidden=32, train_embeddings=False)

    normal = symbolic_condition_binding_metrics(batch, memory_control="normal")
    corrupt_values = symbolic_condition_binding_metrics(batch, memory_control="corrupt_values")
    corrupt_conditions = symbolic_condition_binding_metrics(batch, memory_control="corrupt_conditions")
    random_keys = symbolic_condition_binding_metrics(batch, memory_control="random_keys")
    stress = learned_condition_stress_metrics(reader, batch, memory_control="normal")

    assert normal["condition_symbolic_exact"] == 1.0
    assert corrupt_values["exact_match_available_rate"] == 1.0
    assert corrupt_values["condition_symbolic_exact"] == 0.0
    assert corrupt_conditions["exact_match_available_rate"] == 0.0
    assert corrupt_conditions["condition_symbolic_exact"] == 0.0
    assert random_keys["exact_match_available_rate"] == 0.0
    assert 0.0 <= stress["learned_top1_slot_accuracy"] <= 1.0
    assert 0.0 <= stress["hard_negative_false_positive_rate"] <= 1.0
    assert stress["symbolic_available_rate"] == 1.0


def test_stress_set_symbolic_controls_and_metrics():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=82,
            task="coexisting_stress",
            num_facts=8,
            num_positive=2,
            num_hard_negatives=4,
            similarity_mode="mixed",
            slot_order="random",
            oracle_memory=False,
        )
    )
    batch = dataset.sample_batch(4)
    reader = LearnedSetReader(reader_dim=16, hidden=32, train_embeddings=False)

    normal = symbolic_set_metrics(batch, memory_control="normal")
    corrupt_values = symbolic_set_metrics(batch, memory_control="corrupt_values")
    random_keys = symbolic_set_metrics(batch, memory_control="random_keys")
    stress = learned_set_stress_metrics(reader, batch, memory_control="normal")

    assert normal["symbolic_set_exact"] == 1.0
    assert corrupt_values["symbolic_set_exact"] == 0.0
    assert random_keys["symbolic_set_exact"] == 0.0
    assert 0.0 <= stress["learned_top1_slot_accuracy"] <= 1.0
    assert 0.0 <= stress["missed_positive_rate"] <= 1.0
    assert 0.0 <= stress["extra_false_positive_rate"] <= 1.0
    assert stress["symbolic_available_rate"] == 1.0
