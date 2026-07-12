from pathlib import Path

from scripts.run_memory_model import (
    build_arg_parser,
    make_training_args,
    normalize_summary_row,
    parse_models,
    row_with_run_id,
)


def test_parse_models_allows_single_model():
    assert parse_models("hpm_lite") == ["hpm_lite"]
    assert parse_models("local,hpm_lite") == ["local", "hpm_lite"]


def test_parse_models_deduplicates_preserving_order():
    assert parse_models("hpm_lite,local,hpm_lite") == ["hpm_lite", "local"]


def test_parser_accepts_logging_flags():
    args = build_arg_parser().parse_args(
        [
            "--models",
            "hpm_lite",
            "--log-every",
            "25",
            "--save-step-log",
            "--record-vram",
            "--summary-csv",
            "results/raw/run_summary.csv",
        ]
    )
    assert args.models == ["hpm_lite"]
    assert args.log_every == 25
    assert args.save_step_log is True
    assert args.record_vram is True
    assert args.summary_csv == "results/raw/run_summary.csv"


def test_row_with_run_id_extracts_run_directory_name():
    row = {"run_dir": str(Path("runs") / "memory_model" / "20260627_000000_hpm_lite_kv_seed1")}
    assert row_with_run_id(row)["run_id"] == "20260627_000000_hpm_lite_kv_seed1"


def test_make_training_args_uses_requested_seed_without_offset():
    args = build_arg_parser().parse_args(["--models", "local,hpm_lite", "--seed", "7"])
    assert make_training_args(args, "local").seed == 7
    assert make_training_args(args, "hpm_lite").seed == 7


def test_normalize_summary_row_fills_model_config_and_seed():
    row = {
        "model": "hpm_lite",
        "write_mode": "learned",
        "run_dir": str(Path("runs") / "memory_model" / "abc_hpm_lite_kv_seed0"),
    }
    normalized = normalize_summary_row(row, requested_seed=3, d_model=128, layers=1, heads=4)
    assert normalized["run_id"] == "abc_hpm_lite_kv_seed0"
    assert normalized["seed"] == 3
    assert normalized["d_model"] == 128
    assert normalized["layers"] == 1
    assert normalized["heads"] == 4
    assert normalized["write_mode"] == "learned"


def test_normalize_summary_row_marks_local_memory_fields_as_not_applicable():
    row = {
        "model": "local",
        "write_mode": "oracle",
        "eval_retrieval_top1": 1.0,
        "eval_true_fact_written_rate": 1.0,
        "eval_avg_written_slots": 8.0,
        "run_dir": str(Path("runs") / "memory_model" / "abc_local_kv_seed0"),
    }
    normalized = normalize_summary_row(row, requested_seed=0, d_model=128, layers=1, heads=4)
    assert normalized["write_mode"] == "none"
    assert normalized["eval_retrieval_top1"] == ""
    assert normalized["eval_true_fact_written_rate"] == ""
    assert normalized["eval_avg_written_slots"] == ""
