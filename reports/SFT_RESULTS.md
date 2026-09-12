# Homework 1: SFT results

Evaluated September 12, 2026. Scope: SFT only, Mario persona, GSM8K, Qwen2.5-3B-Instruct. No RLAIF or RLVR was run.

## Summary

Mario SFT produces outputs more similar to the held-out Mario references than the prompted original model. However, its final-answer accuracy is lower than the original model on both validation and test. The plain-GSM8K SFT control also loses accuracy, and the Mario-versus-control differences are inconclusive. These results support a persona-resemblance benefit from SFT, but do not establish that persona training specifically caused the mathematical degradation.

## Data and models

There are 3,000 paired plain/Mario training examples, 500 validation questions, and a separate 500-question subset of official GSM8K test. Training and validation derive from the official training split. The three splits have no exact question overlap. The full 1,319-question test archive was not evaluated in this run.

Both existing released LoRA adapters were used without retraining. The evaluated base revision is `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Adapter weight hashes match those in [the adapter documentation](../adapters/README.md). All 3,000 converted Mario responses match the contextual rewrites after calculator-annotation removal, and their IDs, questions, and final answers align with the plain control data.

All conditions receive the same neutral step-by-step/final-answer formatting instruction. Only the prompted-original condition additionally receives the contextual Mario style instruction. All runs use greedy decoding, BF16, batch size 16, and at most 1,024 new tokens on one NVIDIA L40. Neither checkpoint nor settings were changed after validation outcomes. See [the protocol](../EVALUATION.md) and per-run configurations for exact settings, package versions, full prompts, and hashes.

## Mathematical accuracy

Each cell is based on 500 questions. Format failures count as incorrect. The parser accepts a complete numeric answer after a terminal `####` or inside a balanced final boxed marker, using exact rational equality. It does not accept the prefix of an expression as an answer. This is final-answer accuracy, not proof that every intermediate step is correct.

| Condition | Validation correct | Validation accuracy | Test correct | Test accuracy | Test 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Original, neutral | 426/500 | 85.2% | 365/500 | 73.0% | 69.0-76.8% |
| Original, Mario prompt | 419/500 | 83.8% | 388/500 | 77.6% | 74.0-81.2% |
| Plain-GSM8K SFT, neutral | 372/500 | 74.4% | 326/500 | 65.2% | 61.0-69.4% |
| Mario SFT, neutral | 366/500 | 73.2% | 333/500 | 66.6% | 62.4-70.6% |

| Condition | Validation format success | Test format success | Validation/test token-limit hits |
| --- | ---: | ---: | ---: |
| Original, neutral | 91.4% | 85.8% | 0 / 1 |
| Original, Mario prompt | 93.8% | 92.6% | 0 / 1 |
| Plain-GSM8K SFT, neutral | 99.8% | 99.8% | 0 / 0 |
| Mario SFT, neutral | 99.8% | 99.4% | 0 / 1 |

SFT improves answer-format compliance but does not improve final-answer accuracy here. The measured original-model baseline, not an externally reported 86.7%, is used for every comparison. The validation and test sets are different samples, and their scores must not be treated as measurements on identical questions.

Paired differences below are percentage points, with 95% question-bootstrap intervals (10,000 resamples, seed 42):

| Contrast | Validation difference [95% CI] | Test difference [95% CI] |
| --- | --- | --- |
| Mario SFT minus original | -12.0 [-16.6, -7.4] | -6.4 [-11.0, -2.0] |
| Mario SFT minus plain SFT | -1.2 [-5.6, +3.2] | +1.4 [-3.2, +6.0] |
| Mario SFT minus prompted original | -10.6 [-15.2, -6.0] | -11.0 [-15.6, -6.4] |

## Persona-reference similarity

Character 3-5 gram TF-IDF is fitted only on Mario training responses. Each generated validation answer is compared by cosine similarity to the Mario reference for the same question. The sensitivity analysis removes calculator annotations, final-answer lines, numbers, and operator characters before fitting and scoring. Higher values indicate greater textual resemblance, not necessarily better persona or reasoning.

| Condition | Raw cosine | Math-masked cosine |
| --- | ---: | ---: |
| Original, neutral | 0.3555 | 0.3825 |
| Original, Mario prompt | 0.3593 | 0.3867 |
| Plain-GSM8K SFT, neutral | 0.3458 | 0.3630 |
| Mario SFT, neutral | 0.4021 | 0.4261 |

Mario SFT exceeds prompted original by **0.0428** raw cosine (95% CI 0.0359-0.0499) and **0.0393** math-masked cosine (0.0322-0.0463). Its raw difference from the neutral original is 0.0467 (0.0397-0.0535). The benefit therefore persists after removing explicit mathematical characters, although shared nonnumeric problem vocabulary and response structure can still affect this metric.

No persona-reference similarity is reported on test: we did not generate test persona targets. No independent blinded human or model-judge persona score was collected.

## Example outputs and errors

Three validation examples were sampled with `random.Random(42).sample(range(500), 3)`: zero-based indices 327, 57, and 12. This is a small, unblinded qualitative check, not a representative persona rating. Complete outputs for all four conditions are in the result JSONL files.

- `train-2226`, halving cake sugar over four weeks: all four models answer 3 correctly. Mario SFT opens, "Cake recipes, huh? Reminds me of baking cookies back home - gotta keep track of every ingredient!" and narrates the successive halvings. The prompted original uses ordinary numbered math prose without sustained Mario voice in this example. Mario's closing frosting detail is an embellishment, not a stated problem fact.
- `train-6900`, glass panels in single/double windows: the original, prompted original, and plain SFT answer 80 correctly. Mario SFT uses contextual voice but confuses windows with panels, omits the final multiplication by four, and answers 20. This demonstrates why recognizable style does not establish mathematical correctness.
- `train-4452`, ratios of vehicle counts: both original-model conditions answer 60. Plain SFT skips a dependency and answers 30. Mario SFT reverses a ratio and rounds a fractional truck count to 6. Its voice is weak apart from one accented phrase. Persona quality is not uniform, and both SFT models can lose multi-step reasoning accuracy.

## Hypotheses and limitations

**H0, prompting is weaker than SFT for persona:** supported for the specified held-out textual-resemblance metric and these settings. It is not established as a general statement about persona quality or all possible prompting strategies. The prompt-only instruction was made contextual before evaluation; it contains no few-shot examples.

**H1, persona training drops mathematical accuracy:** supported descriptively relative to the original model. A persona-specific causal penalty is not established: plain SFT also performs worse than the original, and the Mario/plain-SFT difference spans zero on both splits. The test point estimate actually favors Mario SFT over plain SFT by 1.4 percentage points.

Intervals describe question-sampling uncertainty, not variation across training seeds. Historical training arguments/logs are incomplete, so exact training-budget equivalence and training-time truncation cannot be independently reconstructed. Source-solution issues and Astra training-worker exposure to three validation style examples remain disclosed in [the data review](mario_v2_review.md). Only four training examples were supplied to the Sonnet rewrite prefix. No tuning or generation used the held-out test answers as prompts.

The released adapters and outputs satisfy the requested SFT experiment artifacts, but the capability-improvement objective was not achieved. Report the negative result rather than implying that persona learning improved mathematical accuracy.

## Artifacts and verification

- [Training code](../sft.py), [released adapter instructions](../adapters/README.md), [evaluation code](../eval.py), [comparison code](../compare.py), and [run commands](../run_evals.sh).
- [Validation comparisons](../results/comparison_val.json) and [test comparisons](../results/comparison_test.json), including unrounded scores and intervals.
- Eight `results/{base,prompted,control,mario}_{val,test}.jsonl` files contain all 4,000 outputs. Each has a configuration and metrics file; `results/run.log` records execution.
- All 26 result JSON/JSONL files were SHA-256 checked after downloading. Every saved correctness value was independently recomputed locally. All 34 unit/data tests passed on the pod and locally before the full runs.

Compute: an initial RTX 4090 startup attempt was stopped and deleted after connection failures. Evaluation used the L40 at $0.82/hour for about 62 minutes including setup/downloads. Combined runtime-based cost is approximately $0.90, not a settled invoice; billing history was still incomplete at teardown. Both pods were deleted after result verification, and the account had zero remaining pods. No credentials are included in the repository.
