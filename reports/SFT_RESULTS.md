# Homework 1: Mario SFT results

**The Mario adapter helped the model sound more like Mario, but it did not make it better at math.** Its answers were more similar to our Mario references than answers produced with prompting alone. On the test set, however, accuracy fell from 73.0% for the original model to 66.6% for Mario SFT.

Plain-GSM8K SFT also reduced accuracy, to 65.2%. We therefore cannot attribute the drop specifically to learning the persona. The difference between the two SFT models was inconclusive.

We evaluated Qwen2.5-3B-Instruct and the two released LoRA adapters on September 12, 2026. This report covers SFT only.

## Did the model learn Mario's voice?

The results suggest that it did, though not consistently. Mario SFT scored higher than both the original model and the Mario-prompted original when we compared their answers with held-out Mario reference answers.

We measured similarity using character 3-5 gram TF-IDF and cosine similarity. The vectorizer was fitted only on the 3,000 Mario training answers. We then compared each model's answer with the reference for the same validation question. We repeated the comparison after removing numbers, operators, calculator annotations, and final-answer lines to check whether shared math notation explained the difference.

| Model | Similarity | Similarity with math masked |
| --- | ---: | ---: |
| Original | 0.3555 | 0.3825 |
| Original + Mario prompt | 0.3593 | 0.3867 |
| Plain-GSM8K SFT | 0.3458 | 0.3630 |
| Mario SFT | 0.4021 | 0.4261 |

Mario SFT beat prompting by **0.0428** cosine similarity (95% CI: 0.0359-0.0499). With math masked, the difference was **0.0393** (0.0322-0.0463). Compared with the unprompted original, the difference was 0.0467 (0.0397-0.0535).

This supports **H0: persona via prompting is weaker than persona via SFT**, for our prompt and similarity measure. It does not prove that SFT would beat every prompting strategy. Text similarity is also not a direct rating of Mario's voice: shared words and sentence structure can raise the score. We did not collect blinded human or model-judge ratings.

### What the answers looked like

We randomly selected three validation questions for a closer look (seed 42; indices 327, 57, and 12). They show both the improvement in voice and its limits.

**Cake recipe (`train-2226`).** All four models correctly answered that halving 24 cups of sugar over three more weeks leaves 3 cups. Mario SFT began:

> Cake recipes, huh? Reminds me of baking cookies back home - gotta keep track of every ingredient!

It continued narrating the calculations in character. The prompted original used ordinary numbered math prose. Mario's answer also added a detail about frosting that was not in the question.

**Window panels (`train-6900`).** Mario SFT sounded in character but confused windows with glass panels. It skipped the final multiplication by four and answered 20 instead of 80. The other three models were correct.

**Vehicle ratios (`train-4452`).** Both original-model conditions answered 60 correctly. Plain SFT skipped a step and answered 30. Mario SFT reversed a ratio, rounded a fractional truck count, and answered 6. Its voice was weak apart from one accented phrase.

These examples are a small, unblinded check, not a measure of how often the persona works. The full outputs are saved in `results/`.

## What happened to math accuracy?

Both SFT models performed worse than the original model. Mario prompting, without SFT, had the highest test accuracy.

| Model | Validation correct | Validation accuracy | Test correct | Test accuracy | Test 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | 426/500 | 85.2% | 365/500 | 73.0% | 69.0-76.8% |
| Original + Mario prompt | 419/500 | 83.8% | 388/500 | 77.6% | 74.0-81.2% |
| Plain-GSM8K SFT | 372/500 | 74.4% | 326/500 | 65.2% | 61.0-69.4% |
| Mario SFT | 366/500 | 73.2% | 333/500 | 66.6% | 62.4-70.6% |

The table below compares models on the same questions. Differences are in percentage points; intervals use 10,000 paired bootstrap resamples with seed 42.

| Comparison | Validation difference [95% CI] | Test difference [95% CI] |
| --- | --- | --- |
| Mario SFT minus original | -12.0 [-16.6, -7.4] | -6.4 [-11.0, -2.0] |
| Mario SFT minus plain SFT | -1.2 [-5.6, +3.2] | +1.4 [-3.2, +6.0] |
| Mario SFT minus prompted original | -10.6 [-15.2, -6.0] | -11.0 [-15.6, -6.4] |

**H1: persona training drops mathematical accuracy** is supported when comparing Mario SFT with the original model. But the plain-SFT control also lost accuracy. Mario SFT was slightly worse than plain SFT on validation and slightly better on test; both intervals include zero. We found no clear additional accuracy penalty from the persona training itself.

SFT did make the final answers easier to parse. That improvement did not translate into more correct answers.

| Model | Validation format success | Test format success | Validation/test token-limit hits |
| --- | ---: | ---: | ---: |
| Original | 91.4% | 85.8% | 0 / 1 |
| Original + Mario prompt | 93.8% | 92.6% | 0 / 1 |
| Plain-GSM8K SFT | 99.8% | 99.8% | 0 / 0 |
| Mario SFT | 99.8% | 99.4% | 0 / 1 |

## How we ran the comparison

We used 3,000 paired plain/Mario training examples, 500 validation questions, and 500 questions from the official GSM8K test split. Training and validation came from the official training split. There was no exact question overlap across the three sets. We did not evaluate the full 1,319-question test archive or generate Mario reference answers for test.

Both released adapters were used without retraining. All 3,000 converted Mario training answers match our rewrites after removing calculator annotations, and their IDs, questions, and final answers match the plain training data. Adapter hashes match [the release documentation](../adapters/README.md). The evaluated base revision was `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

Every model received the same instruction to solve the problem step by step and end with a machine-readable answer. Only the prompted-original condition also received the Mario style instruction. That instruction asked for contextual voice, not just catchphrases, and contained no examples. Both SFT models were evaluated without a Mario instruction.

We used greedy decoding, BF16, batch size 16, and a 1,024-new-token limit on an NVIDIA L40. We did not change checkpoints or settings after seeing validation results. The [evaluation protocol](../docs/EVALUATION.md) and [run configurations](../provenance/evaluation/) record the exact prompts, versions, and hashes.

Answers were scored by exact numeric equality. The parser accepts a complete number or fraction after a terminal `####` marker or inside a balanced final boxed marker. It rejects incomplete expressions rather than taking their first number. Unparseable answers count as wrong. This checks the final answer, not every reasoning step. We used our measured baseline, not the external 86.7% benchmark. Validation and test contain different questions, so their scores should not be compared as if they were the same sample.

### Limits

This is a single-run study. The confidence intervals describe variation across questions, not across training seeds. Historical training arguments and logs are incomplete, so we cannot fully verify that the two training runs had identical budgets or reconstruct training-time truncation.

The [data review](../provenance/data_review.md) records source-answer issues and the Astra training workers' exposure to three validation style examples. Sonnet received only four training examples. Held-out test answers were not used in generation prompts or tuning.

### Files and checks

The repository includes [training code](../sft.py), [adapter download instructions](../adapters/README.md), [evaluation code](../eval.py), [comparison code](../compare.py), and [run commands](../run_evals.sh). To recover the unrounded comparisons, run `python prepare_data.py` and `python compare.py --split val` / `--split test` from the repository root. The JSON summaries are written to `derived/comparisons/` rather than stored alongside this report.

All 4,000 outputs are saved in eight [result files](../results/). Their configurations and execution log are kept separately in [provenance](../provenance/evaluation/). We verified the 26 original downloaded JSON/JSONL files by SHA-256, recomputed every correctness value locally, and passed all 34 unit/data tests locally and on the pod before the full runs. Redundant generated summaries were later removed from Git; the outputs and historical configurations are unchanged.

The L40 ran for about 62 minutes, including setup and downloads, at $0.82/hour. Including an unsuccessful RTX 4090 startup attempt, estimated compute cost was about $0.90; billing had not settled when we checked. Both pods were deleted after the outputs were verified. No credentials are stored in the repository.

## Conclusion

Mario SFT improved the model's voice, but not reliably on every answer. It also reduced math accuracy relative to the original model. Since plain SFT showed a similar drop, we cannot say the persona caused it. We achieved a stronger Mario persona, but not the assignment's capability-improvement goal.
