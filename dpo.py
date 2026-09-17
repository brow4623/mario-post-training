"""DPO on top of the Mario SFT adapter, from judge-ranked preference pairs.

Step 3 of the RLAIF plan. Starts from the released Mario LoRA adapter and keeps training
those same adapter weights with Direct Preference Optimization, so the result is still one
LoRA adapter over the base model and eval.py can load it unchanged. The frozen reference
model is the Mario SFT model (base + adapter merged), so the KL penalty anchors to the SFT
behaviour rather than to plain Qwen.

Pairs file (data/prefs.jsonl from step 4), one line per question:
  {"id": "train-1458", "prompt": "<question>", "chosen": "<answer>", "rejected": "<answer>", "gold": "108"}
"gold" is optional; when present, "chosen" must pass the verifier unless --skip-verify.
Prompts get the same neutral system instruction as eval.py and sample_candidates.py.

Real run (GPU pod, after downloading adapters/mario per adapters/README.md):
  python dpo.py --prefs data/prefs.jsonl --adapter adapters/mario --output checkpoints/rlaif

Smoke test (any machine, tiny model; --adapter optional, a fresh LoRA is created without it):
  python dpo.py --prefs /tmp/prefs.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --output /tmp/dpo_smoke \
      --max-examples 8 --max-steps 2 --batch-size 2 --grad-accum 1
"""

import argparse
import json
from pathlib import Path

from eval import FORMAT_PROMPT
from mario_data import read_jsonl
from verifier import is_correct


def load_pairs(path, max_examples=None, verify=True):
    """Validate the pairs file and return TRL conversational DPO rows."""
    rows = read_jsonl(path)
    if max_examples:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError(f"{path} has no pairs")
    seen = set()
    for i, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"pair {i}: missing or empty {key!r}")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"pair {i}: chosen and rejected are identical")
        if verify and row.get("gold") is not None and not is_correct(row["chosen"], row["gold"]):
            raise ValueError(f"pair {i} ({row.get('id')}): chosen answer fails the verifier")
        key = (row["prompt"], row["chosen"], row["rejected"])
        if key in seen:
            raise ValueError(f"pair {i}: duplicate pair")
        seen.add(key)
    return [{"prompt": [{"role": "system", "content": FORMAT_PROMPT}, {"role": "user", "content": r["prompt"]}],
             "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--adapter", help="SFT adapter to continue training; omit to start a fresh LoRA (smoke tests)")
    p.add_argument("--beta", type=float, default=0.1, help="KL strength; higher stays closer to the SFT model")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1536, help="prompt + completion tokens")
    p.add_argument("--max-examples", type=int, help="smoke tests")
    p.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--push-to-hub", help="e.g. yourname/qwen3b-mario-rlaif-lora (private)")
    args = p.parse_args()

    pairs = load_pairs(args.prefs, args.max_examples, verify=not args.skip_verify)
    print(f"{len(pairs)} preference pairs from {args.prefs}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    peft_config, ref_model = None, None
    if args.adapter:
        # Keep training the SFT adapter's own weights; freeze a merged copy as the reference.
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        ref_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype),
                                              args.adapter).merge_and_unload()
        ref_model.eval()
    else:
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    cfg = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        bf16=use_cuda,
        gradient_checkpointing=use_cuda,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        dataset_num_proc=1,
    )
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=cfg, processing_class=tok,
                         train_dataset=Dataset.from_list(pairs), peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    print(result.metrics)

    trainer.save_model(args.output)  # adapter weights + config only; loadable by eval.py --adapter
    Path(args.output, "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed adapter to {args.push_to_hub}")


if __name__ == "__main__":
    main()
