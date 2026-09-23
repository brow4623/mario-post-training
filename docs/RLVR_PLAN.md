# Stage 3: RLVR and a combined reward

## Goal

Improve GSM8K accuracy while keeping the Mario voice learned in stages 1 and 2. Run two experiments from the released RLAIF adapter: one that rewards correct answers only, and one that also rewards persona. Compare both with the unchanged RLAIF model.

The starting model scored 66.2% on test math and 2.93/5 on the persona judge. Those are historical results, not guaranteed scores in a new environment. We will rerun the baseline if the evaluation setup changes.

This is a plan, not a completed experiment. No GPU rental or API spending is authorized by this document.

## Experiments

| Condition | Starting weights | Training reward | Purpose |
| --- | --- | --- | --- |
| RLAIF baseline | Released RLAIF adapter | No further training | Measure the starting point |
| Math-only RLVR | Same RLAIF adapter | Correct final answer | Test whether accuracy improves and persona fades |
| Combined reward | Same RLAIF adapter | Correct final answer plus persona | Test whether we can improve math while retaining persona |

Train the two new adapters independently. Do not initialize the combined run from the math-only run: that would confound the reward comparison with extra training. Keep the same training questions, sampling settings, seed, and update budget in both runs. Mario SFT remains background context, not a fourth required run.

## 1. Freeze the data and evaluation

Use the existing 3,000/500/500 train/validation/test split. For a small first experiment, select 512 training questions with seed 42, without filtering by correctness or difficulty. Both runs use that same subset. Give the policy questions only; keep gold answers inside the verifier and never show them to the policy or persona judge.

Check IDs and question text for overlap before training. Keep validation for checkpoint decisions and test for the final comparison. We have already seen the test results from earlier stages, so describe it as a reused benchmark, not a new untouched test set.

Freeze these outcomes before running:

- Primary math outcome: strict final-answer accuracy on GSM8K.
- Primary persona outcome: the existing judge's overall rating, on its 1-5 scale.
- Supporting checks: catchphrase-only rate, rubric dimensions, answer length, and blinded human ratings.

Keep the neutral system prompt and greedy evaluation decoding used in stage 2. Freeze the rubric and judge model rather than changing them after seeing results.

## 2. Implement and test the rewards

Reuse `verifier.py` for exact numeric comparison. Its strict mode accepts explicit final-answer markers and does not use a last-number fallback. Do not change the scoring rule between conditions. Test correct and wrong answers, equivalent fractions and decimals, malformed markers, conflicting answers, trailing prose, and expressions that start with the right number but evaluate differently. Record the existing last-marker behavior explicitly rather than silently introducing a different rule for training.

Define the rewards as follows:

```text
C = 1 if the existing strict verifier accepts the final answer, otherwise 0
P = (persona overall score - 1) / 4

Math-only reward = C
Combined reward = C * (1 + 0.25 * P)
```

Correct answers earn 1 to 1.25 in the combined run; incorrect answers earn 0 regardless of style. This makes correctness the priority while rewarding better persona among correct answers. It combines rule-based and AI-feedback rewards during training, rather than relying only on the earlier DPO stage. The 0.25 weight is a starting design choice, not an established optimum.

Use the existing Haiku 4.5 rubric for `P`. Only correct training completions need a judge call, since the persona term is zero otherwise. For evaluation, judge every answer, including wrong ones. Do not add rewards for Mario catchphrases, answer length, or formatting alone.

Cache each training rating by question, exact answer, model ID, rubric hash, and API configuration. Cache hits must reproduce the same score. A failed judge request must pause or retry the update; it must not become a zero persona rating. Keep a shared spend ledger across retries and resumed jobs. Prompt caching can reduce repeated rubric costs, but output tokens still cost money.

## 3. Build a small GRPO trainer

Use TRL's GRPO trainer with LoRA. GRPO samples several answers per question and learns from their relative rewards. The [official TRL documentation](https://huggingface.co/docs/trl/grpo_trainer) describes custom rewards, reward scaling, and the reference-model penalty. Verify the installed version's API and pin a working environment before the paid run.

Start with these settings, subject to a smoke test:

| Setting | Initial value |
| --- | --- |
| Policy and frozen reference | Released RLAIF model, including its adapter |
| Trainable weights | Continue the existing LoRA weights |
| Training questions | 512, one pass |
| Answers per question | 4 |
| Sampling | Temperature 1.0, top-p 0.95 |
| Maximum completion length | 1,024 tokens, matching the evaluation budget |
| Learning rate | `1e-6` |
| KL penalty coefficient | `0.02` |
| Effective update batch | 16 question groups / 64 completions |
| Seed | 42 |

This is about 32 updates and 2,048 sampled answers per run. Confirm how the pinned trainer counts batches and gradient accumulation before launch. Keep the reference fixed at the full RLAIF model, not base Qwen or Mario SFT. Verify this with parameter hashes and a near-zero initial policy/reference KL.

Compute the combined scalar reward before calculating group advantages. Explicitly disable standard-deviation reward scaling if supported by the pinned version; do not normalize correctness and persona independently, which would change their intended weighting. Log both reward components separately.

Math-only groups with four identical correctness rewards provide no relative learning signal. Track their frequency. If a training-only pilot has fewer than 10 mixed-correctness groups out of 64 questions, reconsider four samples per question before the full run. Make any change symmetrically in both conditions and revise the cost estimate first.

## 4. Run safety and cost checks

First run offline tests for rewards, split isolation, cache keys, API failures, and resume behavior. Then rent one A100-class RunPod GPU for a two-update smoke test. Confirm that only adapter weights change, the reference stays fixed, rewards match hand-checked examples, and both saved adapters reload.

Measure throughput, memory, judge latency, and actual API usage in the pilot. Do not estimate this run from DPO's three-minute training time: generating answers and waiting for judge calls will be substantial parts of GRPO training.

The combined run needs at most 2,048 training judge calls. Evaluating both new models on both splits adds 2,000 calls. A fresh baseline adds another 1,000 if needed. At the previous run's average of about $0.00204 per rating, those totals suggest roughly $8.25-$10.30 for judging before retries or changes in answer length. This is a planning estimate, not a quote or a claim about the current account balance.

Check current RunPod and Anthropic balances before proceeding. Set separate hard spending caps, reserve enough for final evaluation, and get approval for the measured forecast. If funds are short, reduce both training subsets equally before launch rather than finishing training without enough money to evaluate it. Stop and release the GPU after downloading and checking the artifacts.

## 5. Evaluate the three conditions

Generate one answer per question for each condition on all 500 validation and 500 test questions. Use the same model revision, tokenizer, prompt, software, hardware, and decoding settings. Reuse old baseline outputs and judge ratings only if their full configurations match; otherwise regenerate and rescore the baseline.

Use the final checkpoint from each planned one-pass run. Do not choose a checkpoint from training reward or test scores. If validation leads to additional tuning, record every attempt and freeze the final choice before evaluating it on test.

Report these paired comparisons, with 95% question-bootstrap intervals:

| Comparison | Question |
| --- | --- |
| Math-only minus baseline | Did correctness training improve math, and what happened to persona? |
| Combined minus baseline | Did the combined reward improve math while retaining persona? |
| Combined minus math-only | Did the persona reward help, and did it cost accuracy? |

Show accuracy and persona as separate outcomes. Do not use the composite training reward as the headline evaluation metric. Plot accuracy against persona score for the three conditions and show paired differences with uncertainty. Repeat the similar-length check from stage 2.

For an explicit retention criterion, predeclare a maximum acceptable persona loss of 0.20 points out of 5 relative to RLAIF. Call persona retained only if the lower bound of its paired 95% interval exceeds -0.20. Call accuracy improved only if its interval excludes zero on the positive side. This margin is a practical project choice, not a validated standard; report the actual effects and intervals even when these criteria fail.

Have people rate the same 50 randomly selected test questions across all three conditions, with model labels hidden and answer order randomized. That is 150 outputs. Use the existing rubric and report agreement with the judge. If human review is unavailable, say so: optimizing and evaluating with the same judge family can reward judge-specific habits.

## Deliverables and order of work

| Step | Deliverable |
| --- | --- |
| Reward tests and data audit | `rewards.py`, tests, frozen training IDs |
| Trainer and smoke test | `rlvr.py`, pinned dependencies, `run_rlvr.sh` |
| Two training runs | Math-only and combined adapters, release assets and checksums |
| Matched evaluation | Raw outputs and persona scores under `results/rlvr/` |
| Analysis and human review | Comparison tables, figures, blinded rating packet |
| Writeup | `reports/RLVR_RESULTS.md` and a short README update |

Keep run settings, reward records, usage, and checkpoint hashes under `provenance/rlvr/`. Keep regenerable summaries under ignored `derived/`. Publish final adapters as release assets rather than committing checkpoints or model caches.

The result should answer two questions directly: did RLVR improve math, and did adding persona reward preserve Mario's voice better than math-only training? Report an inconclusive or negative result if that is what the comparisons show.
