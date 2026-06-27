from pathlib import Path

from scripts.run_memory_model import parse_models, row_with_run_id, build_arg_parser


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
