# HPM-Lite v2 training integration

This patch wires the v2 core modules into the existing synthetic memory training loop as a real model option:

```bash
python scripts/enable_hpm_lite_v2_training.py
python -m pytest -q tests/test_hpm_v2_modules.py tests/test_hpm_v2_training_integration.py
python -u scripts/run_memory_model.py --models hpm_lite_v2 --seq-len 512 --window 256 --d-model 128 --layers 1 --heads 4 --steps 100 --batch-size 8 --device cuda --memory-null-slot --write-mode learned --learned-writer-teacher-forcing-steps 50 --lambda-writer 0.3 --log-every 25 --save-step-log --record-vram --save-checkpoint false
```

The v2 model keeps the same public forward API as v1 so it can use the current dataset, losses, writer supervision, retrieval metrics, and CSV logging.

This is not a final HPM system yet. It is the first trainable v2 baseline: local attention + blockwise selective recurrent state + fast-weight memory + episodic memory + 4-way router.
