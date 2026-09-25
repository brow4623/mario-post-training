"""GRPO on top of the released RLAIF adapter, with a verifiable or a combined reward.

Stage 3 of docs/RLVR_PLAN.md. The two runs share every setting except the reward:

  math      R = C                      the strict verifier only
  combined  R = C * (1 + 0.25 * P)     the same verifier, scaled by the frozen persona rubric

Both start from the same RLAIF adapter and keep training its LoRA weights, so each run
still produces one adapter over base Qwen that `eval.py --adapter` loads unchanged. Train
them independently: never initialise the combined run from the math-only run.

The frozen KL reference is the whole RLAIF model, not base Qwen and not Mario SFT. Given a
pretrained adapter, TRL 1.13 copies it into a frozen "ref" slot, which is exactly that;
older versions fall back to the adapter-disabled base model, which would anchor to plain
Qwen, so this script pins a merged RLAIF copy itself when the "ref" slot is missing. Either
way it refuses to train against plain Qwen, freezes the reference weights, and records the
parameter hashes and the initial policy/reference KL, which must be ~0.

Real runs (GPU pod, after downloading adapters/rlaif per adapters/README.md):
  python rlvr.py --reward math --adapter adapters/rlaif --output checkpoints/rlvr-math
  python rlvr.py --reward combined --adapter adapters/rlaif --output checkpoints/rlvr-combined \
      --env-file .env --budget-usd 6

Smoke test (training environment, tiny model, fresh LoRA, no API key):
  python rlvr.py --reward math --model Qwen/Qwen2.5-0.5B-Instruct --output /tmp/rlvr_smoke \
      --max-examples 8 --max-steps 2 --batch-size 4 --grad-accum 1 --max-completion-length 64
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

from eval import FORMAT_PROMPT, sha256
from make_rlvr_data import FROZEN, load_frozen
import rewards

# Settings the experiment depends on; an installed TRL without them is the wrong version.
REQUIRED_SETTINGS = ("num_generations", "temperature", "top_p", "max_completion_length", "beta")
QWEN_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def build_rows(rows):
    """One GRPO prompt per question. The policy sees the question only; gold stays for the verifier."""
    return [{"prompt": [{"role": "system", "content": FORMAT_PROMPT},
                        {"role": "user", "content": row["question"]}],
             "id": row["id"], "question": row["question"], "gold": row["answer"]} for row in rows]


def check_batching(batch_size, grad_accum, num_generations):
    """GRPO batches are counted in completions: one update covers batch * accumulation of them."""
    completions = batch_size * grad_accum
    if min(batch_size, grad_accum) < 1 or num_generations < 2:
        raise ValueError("Batch sizes must be positive and at least 2 answers per question are needed")
    if batch_size % num_generations or completions % num_generations:
        raise ValueError(f"{num_generations} answers per question must divide both the device batch "
                         f"({batch_size}) and the {completions} completions per update")
    return {"completions_per_update": completions, "questions_per_update": completions // num_generations}


def scale_rewards_value(config_cls):
    """Turn off standard-deviation reward scaling, whichever API the installed TRL exposes.

    Older versions take a boolean; newer ones take "group"/"batch"/"none". Returns None when
    the version has no such setting, which the manifest records rather than hiding.
    """
    field = next((f for f in dataclasses.fields(config_cls) if f.name == "scale_rewards"), None)
    if field is None:
        return None
    if isinstance(field.default, bool):
        return False
    if isinstance(field.default, str):
        return "none"
    return "none" if "str" in str(field.type) else False


def config_kwargs(config_cls, requested, required=REQUIRED_SETTINGS):
    """Keep the settings this TRL version understands; refuse to run without the essential ones."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    missing = sorted(name for name in required if name not in names)
    if missing:
        raise RuntimeError(f"The installed TRL has no {', '.join(missing)}; pin the version in requirements-rlvr.txt")
    dropped = sorted(name for name in requested if name not in names)
    return {name: value for name, value in requested.items() if name in names}, dropped


def initial_kl(log_history):
    """The KL logged for the first update, before any weights moved."""
    return next((entry["kl"] for entry in log_history if "kl" in entry), None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward", choices=rewards.KINDS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frozen", default=str(FROZEN), help="frozen training questions from make_rlvr_data.py")
    p.add_argument("--data", help="override the source file recorded in the frozen list")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision", help=f"exact base revision; defaults to {QWEN_REVISION} for Qwen 3B")
    p.add_argument("--adapter", help="RLAIF adapter to continue training; omit for a fresh LoRA (smoke tests)")
    p.add_argument("--persona-weight", type=float, default=rewards.PERSONA_WEIGHT, help="w in C * (1 + w * P)")
    p.add_argument("--num-generations", type=int, default=4, help="answers sampled per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-completion-length", type=int, default=1024, help="matches the evaluation budget")
    p.add_argument("--lr", type=float, default=1e-6, help="0 measures the starting policy without training it")
    p.add_argument("--beta", type=float, default=0.02, help="KL penalty against the frozen RLAIF reference")
    p.add_argument("--batch-size", type=int, default=8, help="completions per device step")
    p.add_argument("--grad-accum", type=int, default=8, help="device steps per update")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    p.add_argument("--max-examples", type=int, help="smoke tests: the first N frozen questions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-initial-kl", type=float, default=2e-3,
                   help="the policy must start at the reference; bf16 merge noise only")
    p.add_argument("--run-name", help="records directory under provenance/rlvr; defaults to the output name")
    p.add_argument("--judge-model", default=rewards.judge.MODEL)
    p.add_argument("--judge-max-tokens", type=int, default=rewards.JUDGE_MAX_TOKENS)
    p.add_argument("--judge-workers", type=int, default=8, help="concurrent persona calls inside one batch")
    p.add_argument("--budget-usd", type=float, help="hard ceiling on judge spend, shared with earlier attempts")
    p.add_argument("--env-file", help="dotenv file holding ANTHROPIC_API_KEY (combined reward only)")
    p.add_argument("--cache", help=f"persona rating cache; default {rewards.PROVENANCE / 'persona_cache.jsonl'}")
    p.add_argument("--sample-completions", type=int, default=32, help="completions saved with full text for review")
    p.add_argument("--push-to-hub", help="e.g. yourname/qwen3b-mario-rlvr-math-lora (private)")
    args = p.parse_args(argv)
    if args.reward == "math" and (args.budget_usd or args.env_file):
        p.error("The math reward makes no judge calls; drop --budget-usd and --env-file")
    if args.revision is None:
        args.revision = QWEN_REVISION if args.model == "Qwen/Qwen2.5-3B-Instruct" else "main"
    args.run_name = args.run_name or Path(args.output).name
    return args, check_batching(args.batch_size, args.grad_accum, args.num_generations)


def main():
    args, batching = parse_args()
    output, records = Path(args.output), rewards.PROVENANCE / args.run_name
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory is not empty; use a new directory")
    if (records / "rewards.jsonl").exists():
        raise SystemExit(f"{records} already holds reward records; use a new --run-name")
    selected, frozen = load_frozen(args.frozen, args.data)
    rows = build_rows(selected[:args.max_examples])
    print(f"{len(rows)} training questions from {frozen['source']} (seed {frozen['seed']}), "
          f"{batching['questions_per_update']} questions and {batching['completions_per_update']} completions per update")
    per_update = batching["questions_per_update"]
    if len(rows) < per_update:
        raise SystemExit(f"{len(rows)} questions cannot fill one update of {per_update}; "
                         f"lower --batch-size or --grad-accum for a smoke test")
    if len(rows) % per_update:
        print(f"note: {len(rows) % per_update} questions fall outside the last full update")

    rater, judge_info = None, {}
    if args.reward == "combined":
        # The ledger and the rating cache sit outside the run directory, so a resumed or
        # restarted run shares one spend total and pays only for answers it has not rated.
        rater, judge_info = rewards.build_rater(args.env_file, args.judge_model, args.judge_max_tokens,
                                                args.cache, rewards.PROVENANCE / "judge", args.budget_usd)
        print(f"persona judge {judge_info['judge_model']}, ledger {judge_info['ledger']} "
              f"(${judge_info['recorded_spend_usd']:.4f} recorded), {judge_info['cached_ratings']} cached ratings")
    reward = rewards.Reward(args.reward, rater, args.persona_weight, records / "rewards.jsonl",
                            args.judge_workers, records / "samples.jsonl", args.sample_completions)

    import torch
    from datasets import Dataset
    from huggingface_hub import model_info
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    set_seed(args.seed)
    revision = model_info(args.model, revision=args.revision).sha
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
    prompt_tokens = max(len(tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                    return_dict=False)) for row in rows)

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype)
    peft_config = None
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)  # keep training its own weights
    else:
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    requested = {
        "output_dir": args.output, "beta": args.beta, "learning_rate": args.lr,
        "lr_scheduler_type": "constant", "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.epochs, "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.num_generations, "temperature": args.temperature, "top_p": args.top_p,
        "max_prompt_length": prompt_tokens, "max_completion_length": args.max_completion_length,
        # One scalar reward per completion: correctness and persona must not be normalised apart.
        "scale_rewards": scale_rewards_value(GRPOConfig),
        "bf16": use_cuda, "gradient_checkpointing": use_cuda, "logging_steps": 1,
        "save_strategy": "no", "report_to": "none", "seed": args.seed, "dataset_num_proc": 1,
    }
    settings, dropped = config_kwargs(GRPOConfig, requested)
    if dropped:
        print(f"warning: the installed TRL ignores {', '.join(dropped)}")
    if settings.get("scale_rewards") is None:
        print("warning: this TRL version has no scale_rewards setting; check how it normalises group rewards")
    settings = {k: v for k, v in settings.items() if v is not None}
    cfg = GRPOConfig(**settings)

    trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=cfg, processing_class=tok,
                          train_dataset=Dataset.from_list(rows), peft_config=peft_config)
    # Where the KL penalty anchors. Given a pretrained adapter, TRL copies it into a frozen
    # "ref" slot, which is exactly the RLAIF model. Older versions instead disable the
    # adapter, which anchors to plain Qwen, so pin a merged RLAIF copy ourselves.
    peft_model = trainer.accelerator.unwrap_model(trainer.model)
    ref_model = None
    if args.adapter and trainer.ref_model is None and "ref" not in getattr(peft_model, "peft_config", {}):
        ref_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype),
                                              args.adapter).merge_and_unload()
        ref_model.eval()
        ref_model.requires_grad_(False)
        trainer.ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
    refrozen = []
    for name, param in peft_model.named_parameters():
        if ".ref." in name and param.requires_grad:
            param.requires_grad_(False)  # the reference copy must never be updated with the policy
            refrozen.append(name)
    reference = ("frozen copy of the RLAIF adapter" if "ref" in getattr(peft_model, "peft_config", {}) else
                 "merged RLAIF model" if ref_model is not None else "adapter-disabled base model")
    if args.adapter and reference == "adapter-disabled base model":
        raise RuntimeError("The KL reference would be plain Qwen rather than the RLAIF model")
    print(f"KL reference: {reference}; froze {len(refrozen)} reference parameters")
    trainer.model.print_trainable_parameters()
    initial = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}

    def digest_tensors(items):
        digest = hashlib.sha256()
        empty = True
        for name, tensor in items:
            empty = False
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return None if empty else digest.hexdigest()

    def reference_digest():
        if ref_model is not None:
            return digest_tensors(ref_model.state_dict().items())
        return digest_tensors((n, t) for n, t in peft_model.state_dict().items() if ".ref." in n)

    ref_before = reference_digest()
    manifest = {"arguments": vars(args), "questions": len(rows), "batching": batching,
                "max_prompt_tokens": prompt_tokens, "frozen_questions": {k: v for k, v in frozen.items() if k != "ids"},
                "reward": {"kind": args.reward, "persona_weight": args.persona_weight,
                           "formula": "C" if args.reward == "math" else "C * (1 + w * P)", **judge_info},
                "trl_settings": {k: v for k, v in settings.items() if k != "output_dir"}, "ignored_settings": dropped,
                "base_revision": revision, "script_sha256": sha256(__file__),
                "rewards_sha256": sha256("rewards.py"), "verifier_sha256": sha256("verifier.py"),
                "adapter_hashes": {n: sha256(Path(args.adapter) / n) for n in
                                   ("adapter_config.json", "adapter_model.safetensors")} if args.adapter else {},
                "packages": {n: importlib.metadata.version(n) for n in
                             ("torch", "transformers", "datasets", "peft", "accelerate", "trl")},
                "gpu": torch.cuda.get_device_name() if use_cuda else None,
                "reference": reference, "frozen_reference_parameters": len(refrozen),
                "reference_sha256_before": ref_before}
    for directory in (output, records):
        directory.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (records / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result = trainer.train()
    print(result.metrics)
    summary = reward.summary()
    changed = sum(not torch.equal(p.detach().cpu(), initial[n]) for n, p in trainer.model.named_parameters() if n in initial)
    ref_after, start_kl = reference_digest(), initial_kl(trainer.state.log_history)
    checks = {"changed_adapter_tensors": changed, "trained": args.lr > 0,
              "reference": reference, "reference_sha256_after": ref_after,
              "reference_unchanged": None if ref_before is None else ref_before == ref_after,
              "initial_kl": start_kl, "max_initial_kl": args.max_initial_kl,
              # A policy that starts away from its reference is anchored to the wrong model.
              "policy_started_at_reference": None if start_kl is None else abs(start_kl) <= args.max_initial_kl,
              "finite_parameters": all(bool(torch.isfinite(p).all()) for p in trainer.model.parameters() if p.requires_grad),
              "reward_summary": summary}
    (output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (records / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "reward_summary"}, indent=2))
    print(json.dumps(summary, indent=2))
    failures = [name for name in ("reference_unchanged", "policy_started_at_reference", "finite_parameters")
                if checks[name] is False]
    if checks["policy_started_at_reference"] is None:
        print("warning: this TRL version logged no KL, so only the parameter hashes confirm the reference")
    if args.lr > 0 and not changed:
        failures.append("changed_adapter_tensors")
    if failures:
        raise RuntimeError(f"Training verification failed: {failures}. Records are saved under {records}")

    if "ref" in getattr(peft_model, "peft_config", {}):
        peft_model.delete_adapter("ref")  # release one adapter over base Qwen, as eval.py expects
    trainer.save_model(args.output)  # adapter weights and config only; loadable by eval.py --adapter
    (output / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed adapter to {args.push_to_hub}")


if __name__ == "__main__":
    main()
