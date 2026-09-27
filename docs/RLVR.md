# Stage 3: RLVR runbook

The tooling for [the plan](RLVR_PLAN.md) is written and tested offline. **No training has
run, no GPU has been rented, and no judge calls have been paid for**, so there are no
results yet and `reports/RLVR_RESULTS.md` does not exist. This page is what to execute, in
order, once the budget is approved.

Everything below keeps the two conditions matched: same 512 frozen questions, same
sampling, same seed, same update budget, same evaluation. Only the reward differs.

| Step | Input | Output | State |
| --- | --- | --- | --- |
| 1. Freeze the questions | `data/train_mario.jsonl` | `provenance/rlvr/training_questions.json` | done, committed |
| 2. Rewards | `verifier.py`, `judge.py` | `rewards.py` + `tests/test_rewards.py` | done |
| 3. Trainer | `rlvr.py`, `requirements-rlvr.txt` | two LoRA adapters | ready, not run |
| 4. Smoke test and pilot | one GPU pod | group statistics, hand-checked rewards | ready, not run |
| 5. Training runs | RLAIF adapter | `checkpoints/rlvr-math`, `checkpoints/rlvr-combined` | not run |
| 6. Matched evaluation | three adapters | `results/rlvr/` | not run |
| 7. Persona scoring | `judge.py` | `results/rlvr/judge/` | not run |
| 8. Analysis and human review | `analyze_rlvr.py` | tables, figure, blinded packet | not run |
| 9. Writeup | everything above | `reports/RLVR_RESULTS.md` | not written |

## 1. Frozen training questions

`make_rlvr_data.py` selects 512 of the 3,000 training questions with seed 42, with no
filtering by correctness or difficulty, and records the IDs and hashes in
`provenance/rlvr/training_questions.json`. It refuses to run if any selected ID or question
text appears in validation or test. `rlvr.py` reselects on every launch and stops unless the
result still matches that file, so both runs provably train on the same subset.

```bash
python make_rlvr_data.py --check     # recompute and compare; writes nothing
```

The resolved questions go to `derived/rlvr/train_questions.jsonl`, which is not tracked.
Gold answers never reach the policy or the judge: the prompt is the neutral system
instruction from `eval.py` plus the question, and the gold value travels in a separate
dataset column that only the verifier reads.

## 2. Rewards

`rewards.py` defines both rewards on top of the unchanged strict verifier:

```text
C = 1 if verifier.is_correct(answer, gold) else 0
P = (persona overall - 1) / 4

math      R = C
combined  R = C * (1 + 0.25 * P)
```

Correct answers earn 1 to 1.25 in the combined run; wrong answers earn 0 whatever they
sound like. Nothing rewards catchphrases, length, or formatting on their own.

`tests/test_rewards.py` pins the scoring rule rather than discovering it during a paid run:
equivalent fractions and decimals count as correct, a malformed marker or trailing prose
scores zero, `#### 80 + 5` never matches gold 80 by its prefix, and when an answer contains
two `####` markers the last one wins. That last-marker rule is the evaluation behaviour and
training reuses it unchanged.

Persona ratings use the frozen `prompts/mario_judge.md` rubric through Haiku 4.5, the same
configuration the evaluation judge uses. Only correct completions are rated, because the
persona term is multiplied by correctness. Ratings are cached by question, answer, and
judge-configuration hash in `provenance/rlvr/persona_cache.jsonl`; identical answers inside
one batch cost one request; a cache written under a different rubric or model is refused
rather than mixed in. Every attempt lands on the shared ledger in
`provenance/rlvr/judge/attempts.jsonl`, and `--budget-usd` counts that recorded spend plus
in-flight reservations, so a resumed run cannot quietly exceed the cap. A judge request
that keeps failing raises and stops the update: a missing rating never becomes P = 0.

## 3. Trainer

`rlvr.py` runs TRL's GRPO with LoRA. It continues the RLAIF adapter's own weights, so each
run still produces a single adapter over base Qwen that `eval.py --adapter` loads unchanged.

The KL reference is the whole RLAIF model, and the script proves it rather than assuming
it. Given a pretrained adapter, TRL 1.13 copies that adapter into a frozen `ref` slot and
computes the reference log-probabilities from it; older versions instead disable the
adapter, which anchors training to plain Qwen. `rlvr.py` detects which mechanism is in
play, pins a merged RLAIF copy itself when the `ref` slot is missing, refuses to train at
all if the reference would be plain Qwen, and freezes any reference parameter that arrives
trainable. It then records a SHA-256 of the reference weights before and after training,
the count of adapter tensors that changed, and the KL logged for the first update, which
must be ~0 because the policy starts at the reference. Any failure raises after the records
are saved. The released adapter is the trained `default` slot alone: the reference copy is
removed before saving, so `eval.py --adapter` loads it unchanged.

| Setting | Value | Flag |
| --- | --- | --- |
| Training questions | 512, one pass | `--max-examples` |
| Answers per question | 4 | `--num-generations` |
| Sampling | temperature 1.0, top-p 0.95 | `--temperature`, `--top-p` |
| Maximum completion length | 1,024 tokens | `--max-completion-length` |
| Learning rate | `1e-6`, constant | `--lr` |
| KL penalty | `0.02` | `--beta` |
| Effective update batch | 8 x 8 = 64 completions = 16 questions | `--batch-size`, `--grad-accum` |
| Seed | 42 | `--seed` |

That is 32 updates and 2,048 sampled answers per run. GRPO counts batches in completions,
not questions; `check_batching` refuses any combination where the answers per question do
not divide both the device batch and the completions per update.

The combined scalar reward is computed per completion before GRPO forms group advantages,
and standard-deviation reward scaling is switched off (`scale_rewards="none"` on newer TRL,
`False` on older). Correctness and persona are never normalised apart, which would change
their intended weighting. The run manifest records which settings the installed TRL
ignored, and the run stops outright if the installed version lacks one the experiment
depends on.

Both components are logged per completion to `provenance/rlvr/<run>/rewards.jsonl`, and the
first few completions are saved with full text to `provenance/rlvr/<run>/samples.jsonl` for
hand-checking.

Against the pinned `trl==1.13.0`, `transformers==5.17.0`, `peft==0.21.0`, and
`accelerate==1.15.0`, the whole trainer path has been exercised locally on CPU with a tiny
Qwen2 test model: a fresh LoRA, a continued pretrained adapter, and the combined reward
against a stubbed judge. That run confirmed the `ref` slot is detected and left
byte-identical, the initial KL is exactly 0, only the policy adapter changes, the saved
directory holds one adapter, and the reward log records both components. It also showed
that this TRL ignores `max_prompt_length` and `dataset_num_proc`, which the run prints and
the manifest records. What remains untested until the pod: the real 3B model, bf16 on GPU,
and live judge calls.

Smoke test on any machine with the training environment, no GPU or API key:

```bash
python rlvr.py --reward math --model Qwen/Qwen2.5-0.5B-Instruct --output /tmp/rlvr_smoke \
    --max-examples 8 --max-steps 2 --batch-size 4 --grad-accum 1 --max-completion-length 64
```

## 4-6. GPU session

Check the RunPod and Anthropic balances first, set separate hard caps, and reserve enough
for the evaluation before spending anything on training. `bash run_rlvr.sh` installs the
pinned environment, runs the offline tests, then:

1. a two-update smoke test on the real adapter, with a throwaway checkpoint;
2. a 64-question pilot at the real sampling settings with `--lr 0`, so the group statistics
   describe the starting policy without training it. Groups whose four answers are all
   correct or all wrong give GRPO no relative signal; `rewards.py summarize
   --require-mixed-groups 10` stops the script if fewer than 10 of the 64 groups are mixed.
   If it stops, change the answers per question symmetrically in both conditions and redo
   the cost estimate before continuing;
3. the two training runs, independent and identical except for the reward;
4. the matched evaluation of all three conditions on both splits, into `results/rlvr/`.

Start the pod on `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`. PyPI has
since withdrawn torch 2.8.0, so `requirements-rlvr.txt` can no longer install it and it has to
come from the image; `run_rlvr.sh` checks the installed version first and stops with that
message rather than failing several minutes into pip.

`RLVR_SENSITIVITY_WEIGHTS="0.5 0.75 1.0"` adds an optional persona-weight sweep after step 3,
described in [RLVR_SENSITIVITY.md](RLVR_SENSITIVITY.md). It is exploratory, outside the matched
comparison below, and roughly quadruples the judge spend and the evaluation time.

Never initialise the combined run from the math-only adapter: that would confound the
reward comparison with extra training. Measure throughput, memory, judge latency, and
actual API usage during the pilot instead of extrapolating from DPO's three-minute training
time; generation and judge latency dominate GRPO. Download the checkpoints, `results/rlvr/`,
`provenance/evaluation/rlvr/`, and `provenance/rlvr/` before releasing the pod, then release
it.

Release the two adapters the way `adapters/README.md` describes: upload
`adapter_model.safetensors` as a release asset, record its SHA-256, and copy
`adapter_config.json`, `train_args.json`, and `checks.json` into the repository.

## 7. Persona scoring

No GPU needed. Score every answer, including wrong ones, with the same frozen rubric and
the API backend, one file at a time so the scores land under `results/rlvr/judge/`:

```bash
# Pilot first: 10 answers per condition, resumable, then drop --limit for the full run.
for split in val test; do
  for condition in rlvr_baseline rlvr_math rlvr_combined; do
    python judge.py score "results/rlvr/${condition}_${split}.jsonl" \
      --output "results/rlvr/judge/${condition}_${split}.jsonl" \
      --env-file .env --workers 4 --max-tokens 512 --budget-usd 12 \
      --provenance-dir provenance/rlvr/judge-eval
  done
done
```

That is 3,000 ratings: 1,000 per condition, including a freshly judged baseline. At the rate
measured in stage 2, about $0.00204 each, the evaluation costs roughly $6.10, and the
combined run adds at most 2,048 training ratings, about $4.20, fewer in practice because
only correct completions are rated. Together that is roughly $8.25 to $10.30 before retries,
a planning estimate from the previous run's answer lengths rather than a quote. Keep the
training ledger and the evaluation ledger separate, as above, so each has its own cap.

## 8. Analysis and human review

```bash
python analyze_rlvr.py summarize     # tables plus derived/rlvr/summary.json
python analyze_rlvr.py plot          # reports/figures/rlvr_accuracy_persona.png
python analyze_rlvr.py review        # reports/rlvr_review/ + the hidden condition key
```

`summarize` refuses to compare runs whose model, revision, prompt, decoding settings,
package versions, or code hashes differ, or whose ratings come from more than one judge
configuration, or whose ratings do not hash-match the answers they claim to score. It
reports accuracy and persona as separate outcomes with paired 95% question-bootstrap
intervals, for all three comparisons, plus the rubric dimensions, catchphrase-only rate,
answer length, and the similar-length sensitivity check from stage 2.

The predeclared criteria: accuracy improved only when the paired interval excludes zero on
the positive side, persona retained only when its lower bound is above -0.20 points. Report
the effects and intervals even when they fail, and report an inconclusive or negative result
if that is what the comparisons show. The composite training reward is not an evaluation
metric.

`review` builds one blinded packet: 50 test questions, all three answers per question,
labels hidden, order randomised per question, with the key stored separately under
provenance. That is 150 outputs. It refuses to overwrite existing review files. Report
agreement with the judge; if human review does not happen, say so, because optimising and
evaluating with the same judge family can reward judge-specific habits.

## 9. Writeup

`reports/RLVR_RESULTS.md` comes last, after the numbers exist. It should answer two
questions directly: did RLVR improve math, and did adding the persona reward preserve
Mario's voice better than math-only training?
