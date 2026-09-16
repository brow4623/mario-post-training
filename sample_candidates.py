"""Sample several answers per training question from an adapter, for preference-pair building.

Step 2 of the RLAIF plan. Uses the same base model, chat template, neutral system prompt,
and token budget as eval.py, but samples with temperature instead of greedy decoding so
the same question yields different answers. Every answer is run through the verifier.
Only training questions are used; validation and test stay out of preference data.

Real run (GPU pod, after downloading adapters/mario per adapters/README.md):
  python sample_candidates.py --data data/train_mario.jsonl --name mario_train --adapter adapters/mario

Smoke test (any machine, tiny model, no adapter):
  python sample_candidates.py --data data/train_mario.jsonl --name smoke --model Qwen/Qwen2.5-0.5B-Instruct \
      --max-examples 2 --samples 2 --max-new-tokens 48 --output-dir /tmp/smoke --provenance-dir /tmp/smoke

Output: <output-dir>/<name>.jsonl, one line per question:
  {"id", "question", "gold", "answers": [{"text", "pred", "correct", "format_valid",
                                          "generated_tokens", "hit_token_cap"}, ...]}
Resumable: rerunning with the same name and settings continues where it stopped.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

from eval import FORMAT_PROMPT, sha256, validate_rows
from mario_data import read_jsonl
from verifier import extract_answer, is_correct


def select_questions(rows, count, seed):
    """Seeded subset of the training questions, returned in dataset order."""
    if count is None or count >= len(rows):
        return list(rows)
    chosen = set(random.Random(seed).sample(range(len(rows)), count))
    return [row for i, row in enumerate(rows) if i in chosen]


def answer_record(text, length, gold, max_new_tokens, stopped):
    pred = extract_answer(text)
    return {"text": text, "pred": str(pred) if pred is not None else None,
            "correct": is_correct(text, gold), "format_valid": pred is not None,
            "generated_tokens": length, "hit_token_cap": not stopped and length >= max_new_tokens}


def resume(path, rows):
    saved = read_jsonl(path) if path.exists() else []
    if len(saved) > len(rows):
        raise ValueError("Too many saved records")
    for old, source in zip(saved, rows):
        if old["id"] != source["id"] or old["question"] != source["question"] or old["gold"] != source["answer"]:
            raise ValueError("Saved records do not match dataset order/content")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="training file; validation/test are never used here")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--adapter")
    p.add_argument("--num-questions", type=int, default=1000, help="seeded subset of the training questions")
    p.add_argument("--samples", type=int, default=6, help="answers per question")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4, help="questions per batch (times --samples sequences)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-examples", type=int, help="smoke tests: only the first N selected questions")
    p.add_argument("--output-dir", default="data/candidates")
    p.add_argument("--provenance-dir", default="provenance/candidates")
    args = p.parse_args()
    if "val" in Path(args.data).stem or "test" in Path(args.data).stem:
        p.error("Candidates must come from training data only")
    if min(args.samples, args.batch_size, args.max_new_tokens) < 1:
        p.error("Counts must be positive")

    rows = select_questions(validate_rows(read_jsonl(args.data)), args.num_questions, args.seed)[:args.max_examples]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import model_info

    revision = model_info(args.model, revision=args.revision).sha
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    config = {"model": args.model, "revision": revision, "data": args.data, "data_sha256": sha256(args.data),
              "num_questions": len(rows), "samples": args.samples, "seed": args.seed,
              "system_prompt": FORMAT_PROMPT, "do_sample": True, "temperature": args.temperature,
              "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
              "dtype": str(dtype), "adapter": args.adapter,
              "adapter_hashes": {n: sha256(Path(args.adapter) / n) for n in
                                 ("adapter_config.json", "adapter_model.safetensors")} if args.adapter else {},
              "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate")},
              "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
              "script_sha256": sha256(__file__), "verifier_sha256": sha256("verifier.py")}
    output = Path(args.output_dir) / f"{args.name}.jsonl"
    metadata = Path(args.provenance_dir) / f"{args.name}.config.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != config:
            raise ValueError("Run configuration changed; choose a new name")
    elif output.exists():
        raise ValueError("Candidates exist without run metadata")
    else:
        metadata.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    saved = resume(output, rows)
    print(f"{args.name}: {len(saved)}/{len(rows)} questions already sampled on {device}", flush=True)

    if len(saved) < len(rows):
        tok = AutoTokenizer.from_pretrained(args.model, revision=revision)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype).to(device)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            for start in range(len(saved), len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                prompts = [tok.apply_chat_template([{"role": "system", "content": FORMAT_PROMPT},
                           {"role": "user", "content": r["question"]}], tokenize=False,
                           add_generation_prompt=True) for r in batch]
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                torch.manual_seed(args.seed + start)  # deterministic per batch, so resume reproduces the run
                t0 = time.monotonic()
                with torch.inference_mode():
                    gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                         num_return_sequences=args.samples, max_new_tokens=args.max_new_tokens,
                                         pad_token_id=tok.pad_token_id)
                elapsed = time.monotonic() - t0
                tokens = gen[:, enc["input_ids"].shape[1]:].tolist()
                for q, source in enumerate(batch):
                    answers = []
                    for sequence in tokens[q * args.samples:(q + 1) * args.samples]:
                        stop = next((i for i, token in enumerate(sequence) if token in eos), None)
                        length = stop + 1 if stop is not None else len(sequence)
                        text = tok.decode(sequence[:length], skip_special_tokens=True)
                        answers.append(answer_record(text, length, source["answer"], args.max_new_tokens, stop is not None))
                    record = {"id": source["id"], "question": source["question"], "gold": source["answer"],
                              "answers": answers, "batch_seconds_per_question": elapsed / len(batch)}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved.append(record)
                stream.flush()
                print(f"{args.name}: {len(saved)}/{len(rows)} questions, {elapsed / len(batch):.1f}s each", flush=True)

    answers = [a for r in saved for a in r["answers"]]
    summary = {"name": args.name, "questions": len(saved), "answers": len(answers),
               "accuracy": sum(a["correct"] for a in answers) / len(answers),
               "format_success": sum(a["format_valid"] for a in answers) / len(answers),
               "hit_token_cap": sum(a["hit_token_cap"] for a in answers),
               "questions_with_2plus_correct": sum(sum(a["correct"] for a in r["answers"]) >= 2 for r in saved),
               "questions_with_correct_and_wrong": sum(0 < sum(a["correct"] for a in r["answers"]) < len(r["answers"]) for r in saved),
               "questions_all_answers_identical": sum(len({a["text"] for a in r["answers"]}) == 1 for r in saved)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
