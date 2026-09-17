# Week 2: RLAIF runbook

Each step lists its input, what to run, and the file it hands to the next step. Steps 1 through 4 have scripts; steps 5 through 8 depend on the pod runs.

| Step | Owner | Input | Output |
| --- | --- | --- | --- |
| 1. Judge | A | `results/*_val.jsonl` | `results/judge/*_val.jsonl` (done) |
| 2. Sample candidates | C | Mario adapter, `data/train_mario.jsonl` | `data/candidates/mario_train.jsonl` |
| 3. DPO trainer | B | `sft.py`, Mario adapter | `dpo.py` (ready) |
| 4. Preference pairs | A | candidates + judge | `data/prefs.jsonl` (ready) |
| 5. Train | B | `dpo.py`, `data/prefs.jsonl` | RLAIF adapter release |
| 6. Evaluate | C | RLAIF adapter, `run_evals.sh`, `judge.py` | `results/rlaif_*.jsonl`, updated table |
| 7. Gaming check | A | 100 blinded outputs | judge vs human agreement |
| 8. Writeup | C | everything above | report section, talking points |

## Step 1: judge (done)

See the persona judge section of [EVALUATION.md](EVALUATION.md). Baseline validation scores are in `results/judge/`.

## Step 2: sample candidates (GPU pod)

`sample_candidates.py` draws several answers per training question from the Mario SFT adapter and runs the verifier on each. It matches `eval.py` exactly (base model revision, chat template, neutral system prompt, 1,024-token budget) except that it samples at temperature 1.0 instead of decoding greedily. Only training questions are used.

On a RunPod pod with the same image as the evaluations (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`):

```bash
git clone https://github.com/brow4623/hw1-tantia-brown.git && cd hw1-tantia-brown
pip install -r requirements.txt
python prepare_data.py
gh release download adapters-v1 -p qwen3b-gsm8k-mario-lora.safetensors -O adapters/mario/adapter_model.safetensors
sha256sum adapters/mario/adapter_model.safetensors   # must match adapters/README.md
python -u sample_candidates.py --data data/train_mario.jsonl --name mario_train --adapter adapters/mario --batch-size 8
```

Defaults: 1,000 questions chosen with seed 42, 6 answers each, temperature 1.0, top-p 0.95. Expect 1 to 2 hours on one A100-class GPU. The run is resumable: if the pod dies, rerun the same command and it continues from the last saved question. Changing any setting requires a new `--name`.

Before leaving the pod, check the summary it prints. `questions_with_2plus_correct` is the number of questions that can yield a pair with both answers correct, which is what step 4 prefers. If it is below about 600, raise `--samples` to 8 under a new name. `questions_all_answers_identical` should be near zero; if not, temperature is too low for this adapter.

Then commit and push the two files it wrote, `data/candidates/mario_train.jsonl` and `provenance/candidates/mario_train.config.json`. The candidates file is a few megabytes and is meant to be tracked.

Smoke test on any machine, no GPU or adapter needed:

```bash
python sample_candidates.py --data data/train_mario.jsonl --name smoke --model Qwen/Qwen2.5-0.5B-Instruct --max-examples 3 --samples 3 --max-new-tokens 64 --output-dir /tmp/smoke --provenance-dir /tmp/smoke
```

## Step 3: DPO trainer (ready)

`dpo.py` continues training the Mario SFT adapter's own weights with Direct Preference Optimization, so the output is still one LoRA adapter over the base model and `eval.py --adapter` loads it unchanged. The frozen reference for the KL penalty is the Mario SFT model itself (base plus adapter, merged in memory), not plain Qwen. Prompts use the same neutral system instruction as `eval.py` and `sample_candidates.py`.

Input is `data/prefs.jsonl` from step 4, one line per question: `{"id", "prompt", "chosen", "rejected", "gold"}`. The loader refuses empty fields, identical chosen/rejected, duplicates, and any chosen answer that fails the verifier against `gold`.

On the pod, after downloading `adapters/mario`:

```bash
python -u dpo.py --prefs data/prefs.jsonl --adapter adapters/mario --output checkpoints/rlaif
```

Defaults: beta 0.1, learning rate 5e-6, 1 epoch, effective batch 16, max length 1,536 tokens, seed 42. Both models fit on a 40 GB GPU in bf16. Expect well under an hour for 1,000 pairs. Watch two logged numbers: `rewards/accuracies` should climb from 0.5 toward 0.8 or higher, and `rewards/margins` should grow and stay positive. If accuracies stay near 0.5, the pairs are not separable enough (raise the judge-score gap required in step 4). If the loss collapses very fast, lower the learning rate or raise beta.

Then release the adapter the same way as `adapters/README.md` describes: upload `checkpoints/rlaif/adapter_model.safetensors` as a release asset or push to the Hub with `--push-to-hub`, record its SHA-256, and copy `adapter_config.json` plus `train_args.json` into `adapters/rlaif/` in the repository.

Smoke test on any machine (fresh LoRA on the 0.5B model, no adapter needed; `python -m tests.test_dpo --write /tmp/prefs.jsonl` builds a pairs file of Mario versus plain training responses):

```bash
python -m tests.test_dpo --write /tmp/prefs.jsonl
python dpo.py --prefs /tmp/prefs.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --output /tmp/dpo_smoke --max-examples 8 --max-steps 2 --batch-size 2 --grad-accum 1 --max-len 768
```

## Step 4: preference pairs (ready)

Two commands in `make_prefs.py`, plus one judge run between them. Person A runs these once step 2's candidates file is pushed.

```bash
python make_prefs.py flatten data/candidates/mario_train.jsonl
python judge.py score derived/candidates/mario_train.flat.jsonl --backend claude-code --workers 3
python make_prefs.py pairs data/candidates/mario_train.jsonl results/judge/mario_train.flat.jsonl
```

`flatten` writes one judge-ready record per sampled answer with id `<question id>#<answer index>` into `derived/`, which is not tracked. The judge run scores about 6,000 answers; at the rate measured in step 1 that is roughly 50 points of a 5-hour Max window, so plan on two windows or run it overnight. It is resumable, so stopping and restarting later loses nothing. Its output in `results/judge/` is tracked and should be committed.

`pairs` picks, per question, the highest-scoring answer as chosen and the lowest as rejected, then applies three rules. Chosen must pass the verifier. Rejected must pass the verifier too, so the only thing DPO learns from is the persona difference. Chosen must beat rejected by at least 1 point overall. Ties break on the sum of the four criteria, then against the catchphrase-only flag. Questions that fail a rule are dropped and counted by reason in `provenance/prefs/mario_train.summary.json`.

Read the summary before handing off. If `pairs_out` is well under 600, first try `--min-gap 1` with `--allow-wrong-rejected`, which lets a wrong answer be the rejected one and usually recovers many questions at the cost of mixing a math signal into the persona objective. Say which rules were used in the writeup. The output `data/prefs.jsonl` is tracked; `dpo.py` validates it again on load.

Verified end to end on a fake candidates file built from training data (Mario rewrite, plain answer, Mario with a wrong final answer, plain answer with catchphrases stapled on): the judge ranked the rewrite first every time, the wrong-answer Mario was never chosen despite scoring well on persona, and the stapled version was flagged catchphrase-only.
