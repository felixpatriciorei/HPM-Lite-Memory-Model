from types import SimpleNamespace

from hpm_lite.train import run_training
from scripts.run_memory_model import parse_models


def test_runner_accepts_hpm_lite_v2():
    assert parse_models("hpm_lite_v2") == ["hpm_lite_v2"]


def test_hpm_lite_v2_tiny_training_run(tmp_path):
    metrics = run_training(
        SimpleNamespace(
            model="hpm_lite_v2",
            task="kv",
            seq_len=64,
            window=16,
            batch_size=2,
            steps=2,
            eval_every=2,
            eval_batches=1,
            d_model=32,
            layers=1,
            heads=4,
            lr=3.0e-4,
            seed=123,
            device="cpu",
            lambda_ret=0.1,
            lambda_writer=0.1,
            learned_writer_teacher_forcing_steps=1,
            top_k=1,
            memory_null_slot=True,
            null_score_init=0.0,
            memory_control="normal",
            write_mode="learned",
            oracle_memory=True,
            num_facts=4,
            repeated_keys=False,
            similar_values=False,
            distractor_fact_spans=0,
            query_key_noise_only=False,
            fact_order="random",
            out_dir=str(tmp_path),
            save_checkpoint=False,
            log_every=2,
            save_step_log=True,
            record_vram=False,
        )
    )
    assert metrics["model"] == "hpm_lite_v2"
    assert metrics["parameters"] > 0
    assert "eval_answer_exact" in metrics
    assert "eval_retrieval_top1" in metrics
