import torch

from hpm_lite.data import FACT, QUERY, FactRecallConfig, FactRecallDataset
from hpm_lite.noisy_extraction import (
    ConditionCandidateProposerV2,
    ConditionCandidateProposerV3,
    ContextualTupleEdgeScorer,
    LearnedSetExtractorV2,
    LearnedTypedExtractor,
    WriterV3CandidateAssembler,
    _assignment_from_cost,
    writer_quality_metrics,
)
from hpm_lite.structured_readout import symbolic_condition_binding_metrics, symbolic_set_metrics
from hpm_lite.write_modes import apply_write_mode


def test_noisy_conditional_has_typed_pre_query_slots_without_fact_markers():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=91,
            task="noisy_conditional",
            num_facts=8,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=4,
            num_hard_negatives=4,
        )
    )
    batch = dataset.sample_batch(4)

    assert batch["memory_condition_positions"].shape == (4, 8)
    assert not torch.any(batch["input_ids"][:, : batch["answer_positions"].min().item()] == FACT)
    assert symbolic_condition_binding_metrics(batch)["condition_symbolic_exact"] == 1.0

    fact_written, _ = apply_write_mode(batch, "fact_token")
    assert torch.all(fact_written["memory_mask"].sum(dim=1) == 0)

    query_positions = (batch["input_ids"] == QUERY).float().argmax(dim=1).long()
    active = batch["memory_mask"]
    assert torch.all(batch["memory_token_positions"][active] < query_positions[:, None, None].expand_as(batch["memory_token_positions"])[active])
    assert torch.all(batch["memory_condition_positions"][active] < query_positions[:, None].expand_as(batch["memory_condition_positions"])[active])


def test_noisy_coexisting_symbolic_upper_bound_and_oracle_writer_metrics():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=92,
            task="noisy_coexisting",
            num_facts=8,
            num_positive=2,
            noise_level="medium",
            marker_rate=0.5,
            distractor_count=4,
            num_hard_negatives=4,
        )
    )
    batch = dataset.sample_batch(4)

    assert symbolic_set_metrics(batch)["symbolic_set_exact"] == 1.0
    metrics = writer_quality_metrics(batch, batch, has_condition=False)
    assert metrics["slot_f1"] == 1.0
    assert metrics["all_slots_exact"] == 1.0
    assert metrics["post_query_leak_rate"] == 0.0
    assert metrics["answer_token_leak_rate"] == 0.0


def test_learned_typed_extractor_loss_and_predictions_do_not_leak_future_tokens():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=93,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
            num_hard_negatives=2,
        )
    )
    batch = dataset.sample_batch(4)
    extractor = LearnedTypedExtractor(max_slots=4, seq_len=256, has_condition=True, extractor_dim=16, hidden=32)

    loss = extractor.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in extractor.parameters())

    written = extractor.predict_batch(batch)
    metrics = writer_quality_metrics(batch, written, has_condition=True)
    assert metrics["post_query_leak_rate"] == 0.0
    assert metrics["answer_token_leak_rate"] == 0.0


def test_assignment_from_cost_finds_minimum_pairing():
    cost = torch.tensor(
        [
            [5.0, 1.0, 7.0],
            [1.0, 5.0, 7.0],
        ]
    )

    pairs = _assignment_from_cost(cost)

    assert set(pairs) == {(0, 1), (1, 0)}


def test_learned_set_extractor_v2_loss_predictions_and_count_metrics_do_not_leak():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=94,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
            num_hard_negatives=2,
        )
    )
    batch = dataset.sample_batch(4)
    extractor = LearnedSetExtractorV2(
        max_slots=8,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        objectness_threshold=0.5,
    )

    loss = extractor.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in extractor.parameters())

    written = extractor.predict_batch(batch)
    metrics = writer_quality_metrics(batch, written, has_condition=True)
    assert written["memory_mask"].shape == (4, 8)
    assert written["memory_token_positions"].shape == (4, 8, 2)
    assert written["memory_condition_positions"].shape == (4, 8)
    assert "predicted_slot_count" in metrics
    assert "slot_count_accuracy" in metrics
    assert "objectness_accuracy" in metrics
    assert metrics["post_query_leak_rate"] == 0.0
    assert metrics["answer_token_leak_rate"] == 0.0


def test_learned_set_extractor_v2_oracle_modes_decompose_count_and_fields():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=95,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
            num_hard_negatives=2,
        )
    )
    batch = dataset.sample_batch(4)
    extractor = LearnedSetExtractorV2(
        max_slots=8,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        objectness_threshold=0.5,
    )

    none_written = extractor.predict_batch(batch, eval_mode="normal_v2", threshold=1.1)
    assert torch.all(none_written["memory_mask"].sum(dim=1) == 0)

    topk_written = extractor.predict_batch(batch, eval_mode="oracle_count_topk")
    assert torch.all(topk_written["memory_mask"].sum(dim=1) == batch["memory_mask"].sum(dim=1))

    upper_written = extractor.predict_batch(batch, eval_mode="oracle_count_and_fields")
    upper_metrics = writer_quality_metrics(batch, upper_written, has_condition=True)
    assert upper_metrics["slot_count_accuracy"] == 1.0
    assert upper_metrics["slot_f1"] == 1.0
    assert upper_metrics["all_slots_exact"] == 1.0
    assert upper_metrics["post_query_leak_rate"] == 0.0
    assert upper_metrics["answer_token_leak_rate"] == 0.0


def test_writer_v3_oracle_candidates_have_recall_and_do_not_leak():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=96,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
            num_hard_negatives=2,
        )
    )
    batch = dataset.sample_batch(4)
    writer = WriterV3CandidateAssembler(
        max_slots=8,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        objectness_threshold=0.5,
    )

    loss = writer.loss(batch, candidate_k=8, candidate_loss_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in writer.parameters())

    written = writer.predict_batch(batch, candidate_k=8, candidate_mode="oracle_candidates")
    metrics = writer_quality_metrics(batch, written, has_condition=True)
    assert metrics["candidate_key_recall"] == 1.0
    assert metrics["candidate_condition_recall"] == 1.0
    assert metrics["candidate_value_recall"] == 1.0
    assert metrics["all_fields_candidate_recall"] == 1.0
    assert metrics["post_query_leak_rate"] == 0.0
    assert metrics["answer_token_leak_rate"] == 0.0


def test_writer_v3_target_candidate_indices_map_to_gold_positions():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=97,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
        )
    )
    batch = dataset.sample_batch(2)
    writer = WriterV3CandidateAssembler(
        max_slots=4,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        objectness_threshold=0.5,
    )

    candidate_positions, _, _, _ = writer.proposer.candidate_pools(batch, candidate_k=4, mode="oracle_candidates_gold_only")
    target_indices, source_slots = writer._target_candidate_indices(batch, candidate_positions, 0)

    assert len(target_indices) == int(batch["memory_mask"][0].sum().item())
    for indices, slot in zip(target_indices, source_slots):
        assert int(candidate_positions[0, 0, indices[0]].item()) == int(batch["memory_token_positions"][0, slot, 0].item())
        assert int(candidate_positions[0, 1, indices[1]].item()) == int(batch["memory_token_positions"][0, slot, 1].item())
        assert int(candidate_positions[0, 2, indices[2]].item()) == int(batch["memory_condition_positions"][0, slot].item())

    gold_all = writer.predict_batch(batch, candidate_k=4, candidate_mode="oracle_candidates_gold_only", assembly_eval_mode="gold_all_fields")
    metrics = writer_quality_metrics(batch, gold_all, has_condition=True)
    assert metrics["slot_f1"] == 1.0
    assert metrics["all_slots_exact"] == 1.0


def test_contextual_tuple_oracle_candidates_do_not_leak_and_gold_all_is_exact():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=98,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
        )
    )
    batch = dataset.sample_batch(2)
    scorer = ContextualTupleEdgeScorer(
        max_slots=4,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
    )

    loss = scorer.loss(batch, candidate_k=4, candidate_loss_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in scorer.parameters())

    candidate_positions, candidate_mask, _ = scorer.candidate_pools(batch, candidate_k=4, mode="oracle_candidates_gold_only")
    for b in range(batch["input_ids"].size(0)):
        target_tuple_indices, source_slots = scorer._target_tuple_indices(batch, candidate_positions, b)
        assert len(target_tuple_indices) == int(batch["memory_mask"][b].sum().item())
        for flat_idx, slot in zip(target_tuple_indices, source_slots):
            k_count = candidate_positions.size(2)
            key_idx = flat_idx // (k_count * k_count)
            rem = flat_idx % (k_count * k_count)
            cond_idx = rem // k_count
            value_idx = rem % k_count
            assert int(candidate_positions[b, 0, key_idx].item()) == int(batch["memory_token_positions"][b, slot, 0].item())
            assert int(candidate_positions[b, 1, value_idx].item()) == int(batch["memory_token_positions"][b, slot, 1].item())
            assert int(candidate_positions[b, 2, cond_idx].item()) == int(batch["memory_condition_positions"][b, slot].item())
        assert torch.all(candidate_positions[b][candidate_mask[b]] < (batch["input_ids"][b] == QUERY).nonzero(as_tuple=False)[0, 0])

    gold = scorer.predict_batch(batch, candidate_k=4, candidate_mode="oracle_candidates_gold_only", decode_mode="gold_all_fields")
    metrics = writer_quality_metrics(batch, gold, has_condition=True)
    assert metrics["slot_f1"] == 1.0
    assert metrics["all_slots_exact"] == 1.0
    assert metrics["post_query_leak_rate"] == 0.0
    assert metrics["answer_token_leak_rate"] == 0.0


def test_contextual_tuple_debug_examples_are_bounded_and_pre_query():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=99,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=2,
        )
    )
    batch = dataset.sample_batch(2)
    scorer = ContextualTupleEdgeScorer(
        max_slots=4,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
    )

    examples = scorer.debug_examples(batch, candidate_k=4, candidate_mode="oracle_candidates_gold_only", limit=1)

    assert len(examples) == 1
    assert len(examples[0]["top_scored_tuples"]) <= 10
    query_position = int((batch["input_ids"][0] == QUERY).nonzero(as_tuple=False)[0, 0].item())
    for field_items in examples[0]["candidates"].values():
        for item in field_items:
            assert item["pos"] < query_position


def test_contextual_learned_candidate_repair_modes_cover_gold_fields():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=100,
            task="noisy_conditional",
            num_facts=4,
            noise_level="medium",
            marker_rate=0.0,
            distractor_count=4,
        )
    )
    batch = dataset.sample_batch(2)
    scorer = ContextualTupleEdgeScorer(
        max_slots=4,
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
    )

    repaired = scorer.predict_batch(
        batch,
        candidate_k=4,
        candidate_mode="learned_candidates_plus_oracle_missing",
        decode_mode="top_true_count",
    )
    repaired_metrics = writer_quality_metrics(batch, repaired, has_condition=True)
    assert repaired_metrics["candidate_key_recall"] == 1.0
    assert repaired_metrics["candidate_condition_recall"] == 1.0
    assert repaired_metrics["candidate_value_recall"] == 1.0
    assert repaired_metrics["post_query_leak_rate"] == 0.0
    assert repaired_metrics["answer_token_leak_rate"] == 0.0

    oracle_key = scorer.predict_batch(
        batch,
        candidate_k=4,
        candidate_mode="oracle_key_candidates",
        decode_mode="top_true_count",
    )
    key_metrics = writer_quality_metrics(batch, oracle_key, has_condition=True)
    assert key_metrics["candidate_key_recall"] == 1.0


def test_condition_candidate_v2_candidates_are_pre_query_and_report_span_metrics():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=101,
            task="noisy_conditional",
            num_facts=4,
            noise_level="hard",
            marker_rate=0.0,
            distractor_count=4,
            template_mix="paraphrase",
        )
    )
    batch = dataset.sample_batch(2)
    proposer = ConditionCandidateProposerV2(
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        span_condition=True,
        guideline_condition=True,
        simplified_aux_weight=0.5,
        guideline_loss_weight=0.5,
    )

    loss = proposer.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    positions, mask, _, _ = proposer.candidate_pools(batch, candidate_k=8, mode="learned_candidates")
    assert "condition_span_recall" in proposer.last_candidate_debug
    for b in range(batch["input_ids"].size(0)):
        query_position = int((batch["input_ids"][b] == QUERY).nonzero(as_tuple=False)[0, 0].item())
        assert torch.all(positions[b][mask[b]] < query_position)


def test_template_augmentation_uses_extra_training_templates_without_leaking_answer():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=102,
            task="noisy_conditional",
            num_facts=4,
            noise_level="hard",
            marker_rate=0.0,
            distractor_count=2,
            template_mix="simple",
            template_augmentation="heavy",
        )
    )
    batch = dataset.sample_batch(2)
    for b in range(batch["input_ids"].size(0)):
        query_position = int((batch["input_ids"][b] == QUERY).nonzero(as_tuple=False)[0, 0].item())
        assert torch.all(batch["memory_token_positions"][b][batch["memory_mask"][b]] < query_position)
        assert torch.all(batch["memory_condition_positions"][b][batch["memory_mask"][b]] < query_position)


def test_condition_candidate_v3_asymmetric_budget_has_no_future_leakage():
    dataset = FactRecallDataset(
        FactRecallConfig(
            seq_len=256,
            window=64,
            seed=103,
            task="noisy_conditional",
            num_facts=8,
            noise_level="hard",
            marker_rate=0.0,
            distractor_count=8,
            template_mix="paraphrase",
        )
    )
    batch = dataset.sample_batch(2)
    proposer = ConditionCandidateProposerV3(
        seq_len=256,
        has_condition=True,
        extractor_dim=16,
        hidden=32,
        simplified_aux_weight=0.5,
        guideline_loss_weight=0.5,
    )
    loss = proposer.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    positions, mask, _, _ = proposer.candidate_pools(batch, candidate_k=[8, 8, 12], mode="learned_candidates")
    assert positions.shape == (2, 3, 12)
    assert torch.all(~mask[:, 0, 8:])
    assert torch.all(~mask[:, 1, 8:])
    assert "condition_v3_union_recall" in proposer.last_candidate_debug
    for b in range(batch["input_ids"].size(0)):
        query_position = int((batch["input_ids"][b] == QUERY).nonzero(as_tuple=False)[0, 0].item())
        active_positions = positions[b][mask[b]]
        assert active_positions.numel() > 0
        assert torch.all(active_positions < query_position)
