# RLVR results

## Summary

**Neither reward changed the model.** We continued the RLAIF adapter with GRPO under two rewards, a verifier-only one and a combined verifier/persona one, and compared both against the untouched RLAIF baseline on 500 validation and 500 test questions. Test accuracy moved from 66.2% to 67.0% (math-only) and 68.2% (combined); the persona score stayed at 2.93 in all three conditions. Every one of the eighteen paired comparisons we ran has a 95% interval containing zero.

Against the criteria we fixed before training, this is **accuracy not improved, persona retained**. We report it as a null result rather than reading a winner off differences the data cannot distinguish from sampling noise.

The training itself ran correctly, which is what makes the result informative: the reward signal was present, the reference model was verified frozen, and the adapters did change. What the experiment shows is that 32 updates at a learning rate of `1e-6` under a KL penalty of `0.02` is not enough training to move a 3B model's behaviour, whichever of the two rewards drives it.

## The three conditions

All three share the same base model, the same frozen RLAIF starting adapter, the same 512 training questions, the same sampling settings, the same seed, and the same update budget. Only the reward differs.

| Condition | Reward | Judge calls during training | What it tests |
| --- | --- | ---: | --- |
| RLAIF baseline | none, not trained further | 0 | Where stage 2 left the model |
| Math-only RLVR | `R = C` | 0 | Does a verifiable reward improve math? |
| Combined reward | `R = C * (1 + 0.25 * P)` | 1,387 | Does adding persona preserve Mario's voice? |

`C` is 1 when the strict verifier accepts the final answer and 0 otherwise. `P` is the frozen Haiku rubric's overall score mapped from 1–5 onto 0–1. Because the persona term is multiplied by correctness, a wrong answer scores 0 however it sounds, and only correct completions are ever sent to the judge. A correct answer earns between 1.00 and 1.25.

Both trained runs started from the RLAIF adapter independently. The combined run was never initialised from the math-only result, which would have confounded the reward comparison with extra training.

## Neither reward changed math accuracy

| Test accuracy | Value [95% confidence interval] |
| --- | --- |
| RLAIF baseline | 66.2% [62.2%, 70.4%] |
| Math-only RLVR | 67.0% [63.0%, 71.0%] |
| Combined reward | 68.2% [64.2%, 72.2%] |

| Paired difference, test | Accuracy change [95% confidence interval] |
| --- | --- |
| Math-only minus baseline | +0.8 pp [-1.8, +3.4] |
| Combined minus baseline | +2.0 pp [-0.8, +4.8] |
| Combined minus math-only | +1.2 pp [-1.4, +4.0] |

Every interval crosses zero, so none of these differences meets the criterion we predeclared. The validation split tells the same story and adds a useful warning:

| Paired difference, validation | Accuracy change [95% confidence interval] |
| --- | --- |
| Math-only minus baseline | +1.2 pp [-1.2, +3.6] |
| Combined minus baseline | +0.0 pp [-2.4, +2.4] |
| Combined minus math-only | **-1.2 pp** [-4.0, +1.6] |

The combined-minus-math-only contrast is **+1.2 pp on test and -1.2 pp on validation**. The same comparison, run on two splits of the same size, changes sign. That is the clearest evidence in this report that we are looking at noise: a real effect of that size would not reverse direction between splits.

![Accuracy and persona for all three conditions on the test split, with paired differences from the RLAIF baseline.](figures/rlvr_accuracy_persona.png)

## Both rewards preserved the persona

| Test split | Persona overall (1-5) | Catchphrase-only | Words |
| --- | ---: | ---: | ---: |
| RLAIF baseline | 2.930 [2.834, 3.030] | 42.4% | 105 |
| Math-only RLVR | 2.930 [2.838, 3.022] | 41.2% | 104 |
| Combined reward | 2.938 [2.840, 3.036] | 39.8% | 106 |

| Paired persona difference | Test | Validation |
| --- | --- | --- |
| Math-only minus baseline | +0.000 [-0.074, +0.076] | -0.022 [-0.096, +0.052] |
| Combined minus baseline | +0.008 [-0.074, +0.088] | -0.002 [-0.082, +0.078] |
| Combined minus math-only | +0.008 [-0.072, +0.086] | +0.020 [-0.058, +0.100] |

All six lower bounds sit well above the -0.20 retention margin, so persona counts as retained under our criterion. That conclusion is weaker than it looks: persona is retained here because **nothing changed**, not because the persona reward defended it against pressure from the math reward. Math-only training, which has no persona term at all, retained the voice exactly as well.

Answer lengths were within 1.5 words of each other across conditions, so the length-matched sensitivity check adds little; it moves every persona difference by at most 0.03 and leaves all of them spanning zero.

## Why nothing moved

The run was not broken, and we checked that rather than assuming it.

| Verification | Math-only | Combined |
| --- | --- | --- |
| Adapter tensors changed | 504 | 504 |
| Initial policy/reference KL | 0.00116 | 0.00116 |
| Reference SHA-256 after training | unchanged | unchanged |
| Reference identity | frozen RLAIF copy | frozen RLAIF copy |

Both runs recorded an identical reference hash, confirming they trained against the same frozen RLAIF model rather than against plain Qwen, and that hash was unchanged afterwards. The initial KL of 0.00116 is under our 0.002 threshold, confirming each policy genuinely started at its reference. All 504 adapter tensors changed, so optimisation did happen.

The reward also carried real signal. GRPO learns from differences within a group of four answers to the same question, and groups whose answers are all correct or all wrong teach it nothing:

| Group composition, 512 groups | Math-only | Combined |
| --- | ---: | ---: |
| Mixed correct and wrong | 255 | 245 |
| All four correct | 218 | 215 |
| All four wrong | 39 | 52 |
| Mean correctness | 0.704 | 0.677 |

Half the groups were mixed, far above the pilot gate of 10 that would have stopped the run. So the explanation is not a degenerate reward. It is the size of the update budget: **32 optimizer steps**, one pass over 512 questions, at learning rate `1e-6`, with a KL penalty of `0.02` actively pulling the policy back toward the reference. Those settings were chosen to be conservative, and they were conservative enough that the model barely moved.

There is a second, narrower reason the persona term specifically had little room to act. Within a group, a correct answer outscores a wrong one by 1.00, while the entire persona bonus spans at most 0.25. Persona can therefore only steer learning inside the 215 groups where all four answers were already correct, and even there only by the spread of the judge's ratings across four answers to one question. The training log shows a mean persona of 3.03 during the combined run, close to the 2.93 the baseline already scored at evaluation.

## How we ran the experiment

Both runs used TRL's GRPO with LoRA, continuing the RLAIF adapter's own weights so each produced a single adapter over base Qwen. Settings: 512 frozen training questions selected with seed 42 and audited for overlap against both held-out splits, 4 answers sampled per question at temperature 1.0 and top-p 0.95, completions capped at 1,024 tokens, learning rate `1e-6` constant, beta `0.02`, and an effective update of 64 completions covering 16 questions. That is 32 updates and 2,048 sampled answers per run, on one A100-80GB. We did not change any of these after seeing results.

The combined scalar reward is computed per completion before GRPO forms group advantages, and standard-deviation reward scaling is off, so correctness and persona are never normalised apart.

For evaluation we ran all three adapters with the same Qwen2.5-3B-Instruct revision, software environment, neutral system prompt, and greedy decoding, including a freshly re-evaluated RLAIF baseline. Haiku 4.5 (`claude-haiku-4-5-20251001`) then scored all 3,000 answers with the same frozen rubric used during training, seeing each question and answer without the model label or gold answer.

Confidence intervals come from resampling question pairs 10,000 times with seed 42. They capture variation across questions, not across training runs: **we trained one adapter per condition**, so they do not bound run-to-run variance, which is likely the dominant source of the small differences reported above.

Judge spending was $2.83 during combined training and $6.16 for the evaluation ratings, $8.99 in total, tracked on two separate ledgers so each had its own cap. GPU time is not recorded in provenance.

## Limits

- **One run per condition.** Every difference here is within the range that retraining with a different seed could plausibly produce, and the sign flip between splits suggests it is.
- **The judge scored what it trained.** The persona reward and the persona metric come from the same judge family, so the combined run's persona score partly measures how well it optimised its own grader. The blinded human review is the check on that, and it has not been done.
- **Limited resolution.** With 500 questions per split, the paired accuracy intervals are about ±2.5 pp wide. A real improvement smaller than roughly 3 pp would not have been distinguishable from noise here, whatever the training did. Pairing over the same questions buys some sensitivity back: the per-condition intervals are ±4 pp, but the differences we actually test are narrower.
- **The training reward is not an evaluation metric.** The combined run's mean training reward of 0.763 is not comparable to the math run's 0.704; they are computed on different scales.

## Conclusion and next step

Neither RLVR reward produced a detectable change in math accuracy, and both preserved the Mario persona equally, including the one with no persona term. The experiment was executed as planned and verified sound, so the finding is about the training budget rather than about the rewards.

The informative next step is **not** varying the persona weight. Math-only RLVR, which contains no persona reward at all, also failed to move accuracy; until a verifiable reward alone can shift the model at this scale, the shape of the persona term is a second-order question. A weight sweep would most likely return more null results at roughly four times the cost. The [sensitivity sweep](../docs/RLVR_SENSITIVITY.md) is wired up if we want it later.

More promising: raise the update budget, raise the learning rate, or lower `beta`, and re-run the two-condition comparison. The pilot gate, the frozen question set, and the matched evaluation all carry over unchanged.

The remaining check on this run is for people to rate the [50 blinded questions](rlvr_review/review.md), 150 answers with labels hidden and order randomised, since the automated judge both trained and evaluated the combined condition.

The [saved outputs](../results/rlvr/), [judge scores](../results/rlvr/judge/), [training records](../provenance/rlvr/), and [runbook](../docs/RLVR.md) are in the repository.

<details>
<summary>Reproduce the tables and figure without a GPU or API call</summary>

```bash
python analyze_rlvr.py summarize
# Requires Pillow.
python analyze_rlvr.py plot
```

</details>
