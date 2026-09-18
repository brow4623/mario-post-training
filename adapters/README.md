# Trained LoRA adapters

Two LoRA adapters for `Qwen/Qwen2.5-3B-Instruct`, trained with `sft.py` (rank 16, alpha 32, dropout 0.05, all seven attention/MLP projections). Only the small `adapter_config.json` files are tracked here; the 120 MB weight files exceed GitHub's per-file limit and are distributed two ways:

| Adapter | Hugging Face (private) | GitHub release asset | SHA-256 of `adapter_model.safetensors` |
|---|---|---|---|
| control (plain GSM8K) | `brow4623/qwen3b-gsm8k-control-lora` | `qwen3b-gsm8k-control-lora.safetensors` | `9024364a3c3afd7d88644b9828070ec8f7f1ac133e856dcfc96ce1a86a1da16d` |
| mario (Mario persona) | `brow4623/qwen3b-gsm8k-mario-lora` | `qwen3b-gsm8k-mario-lora.safetensors` | `b966ab59a20dee467510a7fc38c1dbac79dec203b0026fe032c8c6b435b6d0db` |

Release: https://github.com/harvard-cs2881f26/brown-ogbogu-tantia/releases/tag/adapters-v1 (original weights from 2026-09-11, copied unchanged to the course repository).

## Using a release asset

Download the weight file into the matching directory as `adapter_model.safetensors`, then pass the directory to `eval.py --adapter`:

```bash
gh release download adapters-v1 -p qwen3b-gsm8k-mario-lora.safetensors -O adapters/mario/adapter_model.safetensors
gh release download adapters-v1 -p qwen3b-gsm8k-control-lora.safetensors -O adapters/control/adapter_model.safetensors
python eval.py --data data/val.jsonl --name mario --adapter adapters/mario
```

Weight files are gitignored (`*.safetensors`), so they never end up in a commit.

## Quality check (2026-09-11)

- All 504 LoRA tensors present in each adapter, correct shapes for Qwen2.5-3B, float32, no NaN/inf, no zero B matrices.
- Adapters are independent trainings: no shared tensors, delta cosine similarity about 0.004.
- Smoke test, first 8 questions of `data/val.jsonl`, greedy, no system prompt: base 3/8 strict (3/8 emit `####`), control 5/8 (8/8), mario 6/8 (8/8, Mario persona in 8/8). Directional only; run `eval.py` on the full split for real numbers.
- Not recorded on the Hub: `train_args.json`. The exact Mario training file is tracked in this repository as `data/train_mario.jsonl`.
