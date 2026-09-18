# RLAIF results

**We trained the DPO adapter, but the completed automatic checks show no clear improvement over Mario SFT.** Test accuracy was 66.2%, compared with 67.0% for a freshly evaluated Mario SFT baseline. Validation text similarity was nearly unchanged. The paired confidence intervals include zero for both measures.

We still need to score the new outputs with the persona rubric. Text similarity alone cannot tell us whether the Mario performance improved. The adapter and all 2,000 answers from the matched comparison are saved.

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

To reproduce the table without a GPU or API call:

```bash
python prepare_data.py
pip install scikit-learn==1.7.2
python compare.py --rlaif --split val --output-dir derived/comparisons/rlaif
python compare.py --rlaif --split test --output-dir derived/comparisons/rlaif
```

Next, run the frozen persona judge on both newly evaluated conditions and do the planned blinded human check. This session had neither an Anthropic credential nor a Claude Code executable available, so we did not run or invent those scores. DPO training is complete; whether it improved the persona remains unresolved.
