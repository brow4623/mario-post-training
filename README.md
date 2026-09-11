# hw1-tantia-brown

Assignment 1: Post-training Qwen2.5-3B-Instruct for persona + STEM capability (SFT, RLAIF, RLVR).

## Current scope

The replacement dataset uses context-aware Mario narration, following the approved [style preview](prompts/mario_examples.md) and [rewrite specification](prompts/mario_rewrite.md). The original phrase-insertion approach was rejected and retired. See [the cached Sonnet runner](SONNET_RUNNER.md) for generation and resume instructions, and `data/generation/summary.json` for recorded progress, cache usage, and cost.

Generation combines 475 saved Astra-medium answers with 3,025 Sonnet 5 low-effort API rewrites. Per-record `rewrite_provenance` distinguishes the two sources. Raw generation records are consolidated in `data/generation/`; partial checkpoints and superseded batches are not included in the current tree.

See [the research plan](RESEARCH_PLAN.md). Prepared configuration: GSM8K, Mario, 3,000 train / 500 validation, seed 42. This branch's dataset work has not run SFT training or model evaluation. The merged repository also includes the training/evaluation pipeline described below; no trained model is included in this merge.

Python 3.11+; preparation and checking use only the standard library. Run commands from this repository's root.

```powershell
python sonnet_rewrite.py --export-only
python assemble_rewrites.py --shards data/generation --publish
python -m unittest discover -s tests -v
python check_answers.py audit --mode rewrite --original data/original/train.jsonl --modified data/mario/train.jsonl --report reports/train_integrity.json
python check_answers.py audit --mode rewrite --original data/original/validation.jsonl --modified data/mario/validation.jsonl --report reports/validation_integrity.json
```

The downloader uses a pinned official OpenAI GSM8K revision. `data/raw` retains original files and the upstream MIT license. `data/manifest.json` records source URLs, hashes, source indices, and generated file hashes. Both selected splits come from official training data. `data/original/test.jsonl` contains all 1,319 official test examples, reserved for final evaluation. Regeneration overwrites generated data files; retain a separate copy before manually editing them.

Each prepared JSONL record contains `id`, unchanged `question`, `answer`, and `messages` (system/user/assistant). Original and Mario variants have identical IDs and questions. The explanatory prose is genuinely rewritten; exact calculator annotations and final `####` lines are retained. Do not concatenate both variants for persona training: use `data/mario/train.jsonl` with assistant-only loss and the original variant for control training and verification. Source caveats and independent human persona/readability review still matter; automated checks do not prove semantic fidelity.

## Check dataset answers

The rewrite audit verifies complete one-to-one pairing, unchanged questions, exact calculator annotations and final lines, chat consistency, and normalized final answers. It deliberately does not require unchanged prose. A failed audit returns a nonzero exit code. To numerically score original or rewritten dataset answers independently:

```powershell
python check_answers.py score --references data/original/train.jsonl --predictions data/original/train.jsonl --field answer --report reports/original_train_answers.json
python check_answers.py score --references data/original/train.jsonl --predictions data/mario/train.jsonl --field answer --report reports/mario_train_answers.json
```

These are reference consistency checks, **not model accuracy** and not proof of persona learning. The checker does not independently solve word problems or validate all intermediate arithmetic.

## Context-aware rewrite batches

The earlier Luna trials are excluded. `data/generation/` contains compact `astra.jsonl` and `sonnet.jsonl` raw records, the exact prompt configuration, usage and attempt logs, source caveats, and ID-ordered final `train_all.jsonl` / `validation_all.jsonl` outputs. `assemble_rewrites.py` validates and pairs them but does not independently certify character quality. See [review limitations](reports/mario_v2_review.md).

```powershell
python assemble_rewrites.py --shards data/generation --split train --allow-partial
python assemble_rewrites.py --shards data/generation --publish
```

Partial review files go to `data/mario_v2/train.preview.jsonl`, never to canonical training files. Publication refuses missing rewrites. Use `check_answers.py audit --mode rewrite` for genuine paraphrases; exact recovery of the old prose is deliberately not required. `mario_data.py` reproduces only the rejected historical baseline and should not be run over the canonical dataset.

## Check actual model outputs

Store completions as JSONL records with exact source IDs and an `output` string. Include only generated assistant text, not the prompt or gold answer. Example schema (illustrative ID and answer, not an actual prediction):

```json
{"id": "gsm8k/train/123", "output": "My reasoning...\n#### 42"}
```

Use the same reference set for base and SFT outputs:

```powershell
python check_answers.py score --references data/original/validation.jsonl --predictions outputs/base_validation.jsonl --report reports/base_validation.json
python check_answers.py score --references data/original/validation.jsonl --predictions outputs/sft_validation.jsonl --report reports/sft_validation.json
```

Scoring uses exact rational-number comparison, accepting properly grouped thousands separators, signed decimals, and fractions. It requires one terminal `#### number` marker; incidental numbers, multiple markers, trailing units, expressions, and malformed values are rejected. Missing predictions count as wrong. Duplicate/unknown IDs and invalid gold answers stop the run. Reports include accuracy, format rate, missing count, and per-example details. Low model accuracy is a valid evaluation result and does not make the scoring command fail.

## Attribution

GSM8K: Cobbe et al. (2021), [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), [official source](https://github.com/openai/grade-school-math). See `data/raw/LICENSE` for the dataset's upstream license. The Mario-style additions are an unofficial educational experiment, not endorsed by Nintendo.
## Training pipeline from main

The scripts below retain their existing data contract from `main`. They expect `response` to contain the worked solution and `answer` to contain only the final number. The context-aware dataset in `data/mario/` instead stores the worked solution in `answer` and includes chat messages. **Do not pass those files directly to `sft.py` or `eval.py`.** Schema conversion and ID/question alignment are required before using the new Mario answers with this pipeline; this conflict-resolution merge does not add that integration.

The two evaluation datasets also differ: `data/test.jsonl` contains a 500-question test subset, while `data/original/test.jsonl` contains all 1,319 official test questions. Keep their results distinct. `verifier.py` and `check_answers.py` use different parsing contracts, so their format-success and accuracy reports are not interchangeable.

## Files

| File | Purpose |
|---|---|
| `verifier.py` | Extracts the final numeric answer (`#### N` or `\boxed{N}`) and checks it for the training/evaluation pipeline. |
| `prepare_data.py` | Downloads GSM8K, writes `data/train_plain.jsonl` (3000), `data/val.jsonl` (500), `data/test.jsonl` (500). |
| `sft.py` | LoRA SFT on a jsonl file, loss on the assistant turn only. |
| `eval.py` | Greedy batched eval of base model + optional adapter + optional system prompt. Appends to `results/metrics.csv`. |
| `mario_prompt.txt` | System prompt for the "persona via prompting" baseline (H0). |

## Data format

One JSON object per line:

```json
{"id": "train-1458", "question": "...", "response": "...worked solution...\n#### 108", "answer": "108"}
```

`data/train_mario.jsonl` must use the same `id`, `question`, and `answer` as `data/train_plain.jsonl`,
with only `response` rewritten in Mario's voice. Every response must still end in `#### <number>`.
`sft.py` runs the verifier over the file and refuses to train if more than 10% of rows fail.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
```

## Smoke test (any machine, ~10 min on CPU)

```bash
python sft.py --data data/train_plain.jsonl --output /tmp/smoke --model Qwen/Qwen2.5-0.5B-Instruct --max-examples 32 --max-steps 4 --batch-size 2 --grad-accum 2 --warmup-steps 1
python eval.py --data data/val.jsonl --name smoke --model Qwen/Qwen2.5-0.5B-Instruct --adapter /tmp/smoke --max-examples 16 --batch-size 8
```

## Real runs (RunPod, one 24 GB GPU)

```bash
export HF_TOKEN=...
python eval.py --data data/val.jsonl --name base
python eval.py --data data/val.jsonl --name base_prompted --system-prompt-file mario_prompt.txt
python sft.py  --data data/train_plain.jsonl --output checkpoints/control --push-to-hub USER/qwen3b-gsm8k-control-lora
python sft.py  --data data/train_mario.jsonl --output checkpoints/mario   --push-to-hub USER/qwen3b-gsm8k-mario-lora
python eval.py --data data/val.jsonl --name control --adapter checkpoints/control
python eval.py --data data/val.jsonl --name mario   --adapter checkpoints/mario
```

Results land in `results/metrics.csv` (committed) and `results/<name>.jsonl` (per-example outputs, gitignored).
