# RLVR Persona Weight Sensitivity Analysis

## Overview

The multi-objective reward function combines math correctness (C) and persona (P) as:

```
R = C * (1 + w * P)
```

where **w** is the persona weight, `--persona-weight` in `rlvr.py`. The default is **w = 0.25**,
which gives a correct answer a bonus of up to 25% according to its persona rating.

That 0.25 was chosen when the reward was designed, not measured. This page describes an
optional sweep over **w ∈ {0.5, 0.75, 1.0}**, alongside the planned w = 0.25 run, to find out
whether the choice matters. It is exploratory and sits outside the matched comparison the
stage 3 plan predeclares; read the caveats at the end before reporting anything from it.

## Reward Scaling Table

For correct answers (C=1), the reward varies with persona overall score and weight:

| Persona Overall | Scaled P | w=0.00 | w=0.25 | w=0.50 | w=0.75 | w=1.00 |
|---|---|---|---|---|---|---|
| 1 (low) | 0.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.25 | 1.000 | 1.062 | 1.125 | 1.188 | 1.250 |
| 3 (average) | 0.50 | 1.000 | 1.125 | 1.250 | 1.375 | 1.500 |
| 4 | 0.75 | 1.000 | 1.188 | 1.375 | 1.562 | 1.750 |
| 5 (high) | 1.00 | 1.000 | 1.250 | 1.500 | 1.750 | 2.000 |

**Key observations:**
- **w = 0.00**: Math-only baseline (no persona signal)
- **w = 0.25**: Default (persona adds up to 25% bonus)
- **w = 0.50**: Persona adds up to 50% bonus
- **w = 0.75**: Persona adds up to 75% bonus
- **w = 1.00**: Persona adds up to 100% bonus (max reward = 2.0 for high persona, correct answer)

Wrong answers always score 0, regardless of w or persona quality.

## Running the Sensitivity Analysis

The sweep is opt-in, because it adds three training runs and three evaluation pairs to the
two-condition matched design in [RLVR_PLAN.md](RLVR_PLAN.md). On the GPU pod described in
[RLVR.md](RLVR.md), set the weights and run the ordinary script:

```bash
RLVR_SENSITIVITY_WEIGHTS="0.5 0.75 1.0" bash run_rlvr.sh
```

With the variable unset, `run_rlvr.sh` runs only the planned `rlvr_math` and `rlvr_combined`
(w = 0.25) conditions. Each listed weight adds one combined training run, trained from the
RLAIF adapter like every other condition, and one evaluation pair.

A single weight on its own is the cheapest way to see whether `w` moves anything at all:

```bash
RLVR_SENSITIVITY_WEIGHTS="1.0" bash run_rlvr.sh
```

## Output Conditions

The evaluation will generate results for:
- `rlvr_baseline` — RLAIF adapter (no training)
- `rlvr_math` — Math-only training. `R = C` is arithmetically the w = 0 case, but it is not run
  through the combined path at all: it makes no judge calls and needs no API key.
- `rlvr_combined` — Combined training with **w=0.25** (default)
- `rlvr_combined_w0_5` — Combined training with **w=0.50**
- `rlvr_combined_w0_75` — Combined training with **w=0.75**
- `rlvr_combined_w1_0` — Combined training with **w=1.00**

Each condition is evaluated on both validation and test splits with greedy decoding.

## Interpreting Results

After persona scoring (step 7) and analysis (step 8), compare accuracy and persona separately
across the four combined runs, as `analyze_rlvr.py summarize` already does for the planned
conditions. Going from w = 0.25 to w = 1.0 quadruples the persona bonus for the same rating,
so it is a wide sweep: if the outputs barely move across it, w is not a knob worth tuning.

**Before reading a trend into the differences, check whether the sweep could have moved
anything at all.** GRPO learns from reward differences *within* a group of four answers to the
same question, and correctness dominates that gap: a correct answer beats a wrong one by 1.0,
while the entire persona range is only `w`. So persona can only steer training inside groups
where all four answers are correct; in mixed groups it is a tiebreaker worth at most `w`
against a gap of 1.0, and in all-wrong groups every reward is 0 regardless.

The pilot already reports the relevant counts: `all_correct_groups` in the
`rewards.py summarize` output is roughly the share of the run where the persona term has room
to act. If that number is small, expect all four weights to produce near-identical adapters,
and say so rather than reporting the sweep as evidence that "the model is robust to w".

Possible readings, in rough order of how much they claim:

- **No visible movement across the sweep.** Either the persona term is too weak to compete
  with correctness given the group composition above, or the judge's ratings do not vary
  enough between the policy's own samples to create a gradient. These are different causes
  with the same signature; the `all_correct_groups` count and the spread of
  `mean_persona_overall` in the reward logs distinguish them.
- **Accuracy falls as w rises.** The persona term is bidding against correctness inside
  all-correct groups and shifting probability mass toward answers that read better but
  generalise worse. Worth reporting with the accuracy intervals attached.
- **Persona rises as w rises, accuracy flat.** The intended behaviour. Note that persona here
  is scored by the same judge family that produced the training signal, so this is partly a
  measure of how well the run optimised its own judge; the blinded human review in step 8 is
  what tests whether the gain is real.
- **Non-monotonic.** With one seed per weight and no replicates, a bumpy curve is at least as
  likely to be run-to-run noise as a real optimum. Do not pick a "best" w from it.

## Testing Locally

`R = C * (1 + w * P)` is linear in `w`, so the existing tests pin it without a separate sweep:
`tests/test_rewards.py` covers the formula at both the default weight and a non-default one,
and `tests/test_rlvr.py` pins that `--persona-weight` reaches the reward unchanged.

```bash
python3 -m unittest tests.test_rewards tests.test_rlvr -q
```

The table above was generated from those same functions; nothing about the sweep needs new
arithmetic tests, only the training runs themselves.

## Cost Implications

**Judge cost does not scale with w.** The weight changes the size of the persona bonus, not
how many completions get rated: every correct completion is sent to the judge in any combined
run, and `w` is applied afterwards. So each combined run costs about the same, roughly the
$4.20 ceiling in [RLVR.md](RLVR.md) for 2,048 completions, and less in practice because wrong
answers are never rated and the cache de-duplicates repeated answers.

The sweep's real cost is that it multiplies the run count:

| | Combined training runs | Evaluation runs (conditions x splits) |
|---|---|---|
| Planned design | 1 | 3 x 2 = 6 |
| With `--sensitivity` | 4 | 6 x 2 = 12 |

So budget roughly 4x the judge spend of one combined run (about $17 before retries), plus
double the evaluation GPU time. `--judge-budget-usd` is a per-run ceiling enforced against
the shared ledger, not a total across runs; set it deliberately and check your Anthropic
balance first.

Two caveats worth stating before spending on this:

- These runs share the seed, questions, and sampling settings, so they differ only in `w`.
  They are **not** independent replicates, and any difference between adjacent weights is
  confounded with ordinary run-to-run variance, which this design does not measure.
- The predeclared criteria in [RLVR.md](RLVR.md) cover the planned `rlvr_math` vs
  `rlvr_combined` comparison. Reading a winner off a four-point sweep is a separate,
  exploratory question; report it as such rather than as a tested hypothesis.
