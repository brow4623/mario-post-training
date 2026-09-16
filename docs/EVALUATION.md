# SFT evaluation protocol

## Scope

Four conditions: original Qwen2.5-3B-Instruct, original plus contextual Mario prompting, plain-GSM8K SFT, and Mario SFT. Released adapters are reused without retraining. All conditions receive the same step-by-step and final-answer-format instruction. The prompted condition additionally receives `mario_prompt.txt`, updated to match the contextual training style rather than listing catchphrases.

Use all 500 validation questions and then all 500 test questions. No test persona references are generated. Greedy decoding, batch size 16, BF16, and 1,024 new tokens are fixed across conditions. Timing pilots use 16 validation questions only and are excluded from the reported runs. Do not select settings or checkpoints using test outcomes.

## Reproduce

Download the adapters using `adapters/README.md`. The RunPod evaluation environment uses the official image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (image digest `sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853`). Its installed PyTorch reports `2.8.0.dev20250319+cu128`, not the stable release implied by the image tag. Exact model revision, installed library versions, code/data/adapter hashes, full system prompts, and hardware are saved in each run's configuration.

```bash
pip install transformers==4.57.1 peft==0.20.0 accelerate==1.10.1 scikit-learn==1.7.2
python prepare_data.py
python -m unittest discover -s tests -q
bash run_evals.sh
```

The commands are run from the repository root. `run_evals.sh` creates a new run under `results/new-run/` and `provenance/evaluation/new-run/`, leaving the submitted outputs untouched. Its first three optional arguments override the output, provenance, and derived-data directories. It writes recalculated comparisons and metrics under `derived/new-run/`.

To recalculate the submitted results without GPU inference, run `python prepare_data.py`, then `python compare.py --split val` and `python compare.py --split test`. These write to `derived/comparisons/`. Historical inference settings and code hashes are retained in `provenance/evaluation/`; see [provenance notes](../provenance/README.md).

Runs append outputs after each batch and resume only when their configuration and ordered source records match. A malformed or truncated JSONL record fails closed rather than being silently dropped. Use a new run name or directory after changing settings. The eight submitted output files and their metadata are kept; inexpensive derived summaries are regenerated.

## Metrics

- Exact numeric answer accuracy uses `verifier.py` equally for all conditions. It parses complete signed numbers, decimals, or fractions, including simple LaTeX fractions. The last explicit `####` or boxed answer is used; malformed expressions cannot be scored by their first number. `####` answers must terminate the completion. Format failures count as wrong, and format success is reported separately. The earlier stricter dataset-integrity parser in `check_answers.py` remains unchanged and is not used to mix evaluation scores.
- Token-limit hits are reported separately. Mathematical accuracy here means final-answer accuracy, not independent verification of every reasoning step.
- Persona-reference resemblance uses character 3-5 gram TF-IDF, fitted only on the 3,000 Mario training responses, and paired cosine similarity to held-out Mario validation references. A math-masked sensitivity analysis removes annotations, final-answer lines, numbers, and operator characters. No validation/test outputs are used to fit the vocabulary or IDF.
- Paired question bootstrap intervals use 10,000 resamples and seed 42. Primary persona comparison is Mario-SFT minus prompted base. Math comparisons are Mario-SFT minus base and Mario-SFT minus plain SFT. These are the user's two directional hypotheses, not a conventional null/alternative pair.

Cosine measures textual resemblance, not a validated persona judgment. Interpret it alongside example outputs and document contrary examples. Intervals reflect question sampling, not training-seed uncertainty. This is a single-run study with incomplete historical training arguments. The 86.7% external benchmark is not substituted for the measured baseline. Source and Astra example-exposure caveats remain in `provenance/data_review.md`.

## Persona judge (Week 2)

`judge.py` scores each answer 1-5 on voice, consistency, naturalness, restraint, and overall, with a catchphrase-only flag and a one-sentence note. The rubric is `prompts/mario_judge.md` and is frozen; its hash is part of every score record's `config_sha256`. The judge sees only the question and the answer text. Gold answers, record ids, condition labels, and file names are never sent. Scores are written to `results/judge/<condition>_<split>.jsonl`, one record per answer, and `judge.py summarize` prints per-condition means, score distributions, and paired bootstrap intervals (10,000 resamples, seed 42) against the first file given.

Two backends exist. The `api` backend sends one answer per request to the Claude API at temperature 0 with a JSON schema; it is the reference protocol and needs an `ANTHROPIC_API_KEY` (pass `--env-file .env`; `.env` is gitignored). The `claude-code` backend runs the same rubric through `claude -p --model haiku` in batches of 20 answers, pooled and shuffled across all input files with seed 42 so every batch mixes conditions. It is billed to a Claude subscription rather than API credits. Its caveats: answers in a batch are visible to each other despite the instruction to rate independently, temperature cannot be set, and Claude Code adds its own context to each call, so scores may shift slightly across Claude Code releases. The Week 2 validation scores were produced with the `claude-code` backend and Haiku 4.5. Per-call usage and any rejected responses are logged to `provenance/judge/attempts.jsonl`.

```bash
python judge.py score results/base_val.jsonl results/prompted_val.jsonl results/control_val.jsonl results/mario_val.jsonl --backend claude-code --workers 3
python judge.py summarize results/judge/mario_val.jsonl results/judge/prompted_val.jsonl results/judge/base_val.jsonl results/judge/control_val.jsonl
```
