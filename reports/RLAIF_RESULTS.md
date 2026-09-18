# RLAIF results

**DPO improved Mario persona scores under the frozen rubric judge, without a clear change in math accuracy.** On the test set, the judge's overall rating rose from 2.58 to 2.93 out of 5, a paired gain of 0.35 points [95% CI: 0.24, 0.46]. The earlier text-similarity measure missed this improvement.

This supports learning the judge's persona preferences, not independent confirmation of better character performance. The same judge family supplied the training preferences. Human review is still pending, and a test score of 2.93/5 leaves substantial room for improvement.

## Persona evaluation

![Paired persona-score improvements, including the similar-length sensitivity check.](figures/rlaif_persona_judge.png)

We scored all 2,000 saved answers with `claude-haiku-4-5-20251001`, using the unchanged rubric and identical API settings for both conditions. The judge saw the question and answer, not the model label or gold answer. These individual API ratings should not be compared directly with the older Claude Code batch ratings; the backend and context differ.

| Measure | Mario SFT | RLAIF | RLAIF minus SFT [95% CI] |
| --- | ---: | ---: | --- |
| Validation overall score (1-5) | 2.632 | 3.122 | +0.490 [0.376, 0.604] |
| Test overall score (1-5) | 2.582 | 2.930 | +0.348 [0.238, 0.458] |
| Validation catchphrase-only rate | 55.8% | 37.0% | -18.8 pp [-24.0, -13.4] |
| Test catchphrase-only rate | 54.8% | 42.2% | -12.6 pp [-18.0, -7.2] |

Voice, consistency, naturalness, and restraint also improved on both splits. Overall score was the primary outcome; the other rubric dimensions are secondary checks. On test, RLAIF scored higher on 154 questions, tied on 278, and scored lower on 68. All rates and scores in this section are the automated judge's assessments.

**The gain was not confined to substantially longer answers.** RLAIF used more words on average: 100 versus 86 on validation, and 105 versus 87 on test. Among pairs with word counts within 10% of each other, its overall-score advantage remained positive: +0.293 [0.144, 0.448] on 181 validation pairs and +0.196 [0.043, 0.342] on 184 test pairs. This subset check does not prove that length played no role; it selects only part of the data and does not remove every possible judge bias.

The 2,000 API calls cost an estimated $4.08 from recorded token usage at [Haiku 4.5 standard rates](https://platform.claude.com/docs/en/about-claude/pricing). All returned valid scores, with no recorded HTTP errors. No GPU was needed for judging.

## Comparison with Mario SFT

| Measure | Mario SFT | RLAIF | RLAIF minus SFT [95% CI] |
| --- | ---: | ---: | --- |
| Validation accuracy | 75.2% (376/500) | 76.6% (383/500) | +1.4 pp [-2.4, +5.4] |
| Test accuracy | 67.0% (335/500) | 66.2% (331/500) | -0.8 pp [-4.6, +3.0] |
| Validation reference similarity | 0.4000 | 0.4015 | +0.0015 [-0.0042, +0.0072] |
| Validation similarity, math masked | 0.4228 | 0.4224 | -0.0004 [-0.0057, +0.0048] |

Accuracy differences are in percentage points (pp). We evaluated both adapters on the same 500 validation and 500 test questions, using the same base revision, neutral system prompt, library versions, A100 GPU, and greedy decoding. We reran Mario SFT because the earlier SFT report used a different software environment. Use this matched baseline for the DPO comparison; the original SFT results remain unchanged in their separate report.

The intervals use 10,000 paired question-bootstrap samples with seed 42. Similarity uses character 3-5 gram TF-IDF fitted on the Mario training answers, followed by cosine similarity against same-question validation references. It measures textual resemblance, not a direct persona rating. These intervals describe uncertainty across questions, not across training runs. We trained only one adapter with one seed.

## Training

We continued the released Mario LoRA adapter with DPO on 773 judge-ranked preference pairs. Both answers in each pair had the correct final numeric answer, so the preference favored persona rather than answer correctness. The frozen reference was the original Mario SFT model. We did not change settings after looking at evaluation results.

The run used one epoch, learning rate `5e-6`, beta `0.1`, effective batch size 16, and seed 42. It completed 49 optimizer steps in 192.6 seconds, with mean training loss 0.6363. All 504 adapter tensors changed, all trainable parameters remained finite, and the reference-model hash stayed identical. The adapter reloaded successfully for evaluation. These checks establish that training ran, not that the resulting persona is better.

The RunPod session lasted about 36.8 minutes, including setup, the smoke test, training, and evaluation. At $1.59/hour, the estimated GPU charge was $0.98, excluding storage and any billing adjustments. We downloaded and hash-checked the outputs, then deleted the pod. No pods remained afterward.

## Files and next steps

- [Adapter download instructions](../adapters/README.md#rlaif-adapter) and [release](https://github.com/harvard-cs2881f26/brown-ogbogu-tantia/releases/tag/rlaif-v1).
- [Saved outputs](../results/): `rlaif_val.jsonl`, `rlaif_test.jsonl`, and the two `mario_rlaif_baseline_*.jsonl` files.
- [Training records](../provenance/training/rlaif/README.md), [evaluation configurations](../provenance/evaluation/), and [runbook](../docs/RLAIF.md).
- [Judge scores](../results/judge/), [judge protocol](../docs/PERSONA_EVALUATION.md), and [blinded human-review packet](persona_review/review.md). The packet has 50 question pairs and 100 blank rating rows; no human scores have been collected yet.

To reproduce the table without a GPU or API call:

```bash
python prepare_data.py
pip install scikit-learn==1.7.2
python compare.py --rlaif --split val --output-dir derived/comparisons/rlaif
python compare.py --rlaif --split test --output-dir derived/comparisons/rlaif
python evaluate_persona.py summarize
# Optional: regenerate the figure (requires Pillow).
python evaluate_persona.py plot
```

The remaining check is blinded human review. The automated results now support improved persona performance under this judge, while the math comparison remains inconclusive. We should not treat either result as proof of broad character quality or unchanged mathematical capability.
