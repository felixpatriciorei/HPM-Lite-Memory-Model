from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"Could not find expected text in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_train() -> None:
    path = ROOT / "hpm_lite" / "train.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from .model import HpmLiteConfig, HpmLiteModel",
        "from .model import HpmLiteConfig, HpmLiteModel\nfrom .hpm_v2_model import HpmLiteV2Config, HpmLiteV2Model",
        path=path,
    )

    text = text.replace(
        'parser.add_argument("--model", choices=["local", "recurrent", "epmem", "hpm_lite", "hebbian"], default="local")',
        'parser.add_argument("--model", choices=["local", "recurrent", "epmem", "hpm_lite", "hpm_lite_v2", "hebbian"], default="local")',
    )
    text = text.replace('"hpm_lite", "hebbian"', '"hpm_lite", "hpm_lite_v2", "hebbian"')

    pattern = re.compile(
        r"def make_model\(args: argparse\.Namespace, device: torch\.device\) -> HpmLiteModel:\n"
        r".*?\n    return HpmLiteModel\(config\)\.to\(device\)",
        re.DOTALL,
    )
    replacement = '''def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model == "hpm_lite_v2":
        config = HpmLiteV2Config(
            model_type="hpm_lite_v2",
            vocab_size=VOCAB_SIZE,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            window=args.window,
            max_seq_len=max(2048, args.seq_len + 1),
            block_size=args.window,
            use_null_slot=args.memory_null_slot,
            null_score_init=args.null_score_init,
            use_learned_writer=args.write_mode == "learned",
        )
        return HpmLiteV2Model(config).to(device)

    config = HpmLiteConfig(
        model_type=args.model,
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        window=args.window,
        max_seq_len=max(2048, args.seq_len + 1),
        use_null_slot=args.memory_null_slot,
        null_score_init=args.null_score_init,
        use_learned_writer=args.write_mode == "learned",
    )
    return HpmLiteModel(config).to(device)'''
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not patch make_model in {path}")

    path.write_text(text, encoding="utf-8")


def patch_runner() -> None:
    path = ROOT / "scripts" / "run_memory_model.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace('VALID_MODELS = {"local", "hpm_lite"}', 'VALID_MODELS = {"local", "hpm_lite", "hpm_lite_v2"}')
    text = text.replace(
        'write_mode=args.write_mode if model == "hpm_lite" else "oracle",',
        'write_mode=args.write_mode if model in {"hpm_lite", "hpm_lite_v2"} else "oracle",',
    )
    text = text.replace('if "local" in by_model and "hpm_lite" in by_model:', 'if "local" in by_model and ("hpm_lite" in by_model or "hpm_lite_v2" in by_model):')
    text = text.replace('hpm = by_model["hpm_lite"]', 'hpm = by_model.get("hpm_lite", by_model.get("hpm_lite_v2"))')
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_train()
    patch_runner()
    print("enabled hpm_lite_v2 in train.py and scripts/run_memory_model.py")


if __name__ == "__main__":
    main()
