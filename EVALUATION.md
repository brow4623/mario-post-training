# SFT evaluation protocol

## Scope

Four conditions: original Qwen2.5-3B-Instruct, original plus contextual Mario prompting, plain-GSM8K SFT, and Mario SFT. Released adapters are reused without retraining. All conditions receive the same step-by-step and final-answer-format instruction. The prompted condition additionally receives `mario_prompt.txt`, updated to match the contextual training style rather than listing catchphrases.

Use all 500 validation questions and then all 500 test questions. No test persona references are generated. Greedy decoding, batch size 16, BF16, and 1,024 new tokens are fixed across conditions. Timing pilots use 16 validation questions only and are excluded from the reported runs. Do not select settings or checkpoints using test outcomes.

## Reproduce

Download the adapters using `adapters/README.md`. The RunPod evaluation environment uses the official image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (image digest `sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853`). Its installed PyTorch reports `2.8.0.dev20250319+cu128`, not the stable release implied by the image tag. Exact model revision, installed library versions, code/data/adapter hashes, full system prompts, and hardware are saved in each run's configuration.

```bash
pip install transformers==4.57.1 peft==0.20.0 accelerate==1.10.1 scikit-learn==1.7.2
python -m unittest discover -s tests -q
bash run_evals.sh
```

Runs append outputs after each batch and resume only when their configuration and ordered source records match. A malformed or truncated JSONL record fails closed rather than being silently dropped. Use a new run name after changing settings. The eight run files and their metadata are submission artifacts, not temporary checkpoints.

## Metrics

- Exact numeric answer accuracy uses `verifier.py` equally for all conditions. It parses complete signed numbers, decimals, or fractions, including simple LaTeX fractions. The last explicit `####` or boxed answer is used; malformed expressions cannot be scored by their first number. `####` answers must terminate the completion. Format failures count as wrong, and format success is reported separately. The earlier stricter dataset-integrity parser in `check_answers.py` remains unchanged and is not used to mix evaluation scores.
- Token-limit hits are reported separately. Mathematical accuracy here means final-answer accuracy, not independent verification of every reasoning step.
- Persona-reference resemblance uses character 3-5 gram TF-IDF, fitted only on the 3,000 Mario training responses, and paired cosine similarity to held-out Mario validation references. A math-masked sensitivity analysis removes annotations, final-answer lines, numbers, and operator characters. No validation/test outputs are used to fit the vocabulary or IDF.
- Paired question bootstrap intervals use 10,000 resamples and seed 42. Primary persona comparison is Mario-SFT minus prompted base. Math comparisons are Mario-SFT minus base and Mario-SFT minus plain SFT. These are the user's two directional hypotheses, not a conventional null/alternative pair.

Cosine measures textual resemblance, not a validated persona judgment. Interpret it alongside example outputs and document contrary examples. Intervals reflect question sampling, not training-seed uncertainty. This is a single-run study with incomplete historical training arguments. The 86.7% external benchmark is not substituted for the measured baseline. Source and Astra example-exposure caveats remain in `reports/mario_v2_review.md`.
