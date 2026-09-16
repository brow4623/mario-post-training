# Week 2: RLAIF runbook

Each step lists its input, what to run, and the file it hands to the next step. Steps 1 and 2 are done or ready; later steps are added as their scripts land.

| Step | Owner | Input | Output |
| --- | --- | --- | --- |
| 1. Judge | A | `results/*_val.jsonl` | `results/judge/*_val.jsonl` (done) |
| 2. Sample candidates | C | Mario adapter, `data/train_mario.jsonl` | `data/candidates/mario_train.jsonl` |
| 3. DPO trainer | B | `sft.py`, Mario adapter | `dpo.py` |
| 4. Preference pairs | A | candidates + judge | `data/prefs.jsonl` |
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

## Step 4 preview: what the judge needs from step 2

`judge.py` scores flat `{id, question, <text field>}` records. Step 4 flattens each candidate answer into its own record with id `<question id>#<answer index>`, scores them all through the judge, then picks chosen and rejected per question. That flattening script arrives with step 4.
