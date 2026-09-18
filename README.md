# Mario post-training on GSM8K

**RLAIF training is complete, but the automatic comparison shows no clear improvement over Mario SFT.** Test accuracy was 66.2% versus 67.0% for the matched SFT baseline, and text similarity was nearly unchanged. Persona-rubric judging is still pending. See the [RLAIF report](reports/RLAIF_RESULTS.md), [adapter release](https://github.com/harvard-cs2881f26/brown-ogbogu-tantia/releases/tag/rlaif-v1), and [runbook](docs/RLAIF.md).

## Original SFT experiment

**Fine-tuning helped Qwen2.5-3B-Instruct sound more like Mario, but it did not improve its math.** Mario SFT scored higher than prompting on similarity to our Mario reference answers. Its test accuracy fell from 73.0% for the original model to 66.6%.

Plain-GSM8K SFT also lost accuracy, reaching 65.2%. We found no clear additional accuracy penalty from learning the persona rather than from SFT more generally.

For the SFT stage of Homework 1, we used 3,000 training examples, 500 validation questions, and 500 test questions. The later DPO run used 773 preference pairs. No RLVR has run.

## SFT results

| Model | Validation accuracy | Test accuracy | Mario-reference similarity |
| --- | ---: | ---: | ---: |
| Original | 85.2% | 73.0% | 0.3555 |
| Original + Mario prompt | 83.8% | 77.6% | 0.3593 |
| Plain-GSM8K SFT | 74.4% | 65.2% | 0.3458 |
| Mario SFT | 73.2% | 66.6% | 0.4021 |

Similarity is measured on validation answers using character n-gram TF-IDF cosine similarity. Mario SFT's advantage remained after masking math notation. This measures textual resemblance, not a blinded rating of the persona. The [full report](reports/SFT_RESULTS.md) includes confidence intervals, examples, errors, and limitations.

## What to read

- [SFT report](reports/SFT_RESULTS.md) and [RLAIF report](reports/RLAIF_RESULTS.md): findings, comparisons, and remaining checks.
- [Trained adapters](adapters/README.md): download links and checksums for Mario SFT, plain-GSM8K SFT, and RLAIF weights. Course staff need access to this private repository and its releases.
- [Model outputs](results/): the original 4,000 SFT evaluation answers and 2,000 answers from the matched RLAIF comparison.
- [Datasets](data/): the paired training files, validation/test files, and original GSM8K source. `train_mario.jsonl` and `train_plain.jsonl` are the files used by the training script.
- [Evaluation protocol](docs/EVALUATION.md): scoring, prompts, settings, and reproduction steps. The [original research plan](docs/RESEARCH_PLAN.md) is kept separately from the completed results.
- [Provenance](provenance/README.md): run configurations, hashes, generation records, logs, and data-review caveats.

## Reproduce the results from saved outputs

Run these commands from the repository root with Python 3.11+. They need no GPU, model downloads, or API key.

```bash
python prepare_data.py
python -m unittest discover -s tests -q
pip install scikit-learn==1.7.2
python compare.py --split val
python compare.py --split test
```

`prepare_data.py` rebuilds the data from the checked-in source and saved rewrites, then checks it against the original hashes. `compare.py` recalculates accuracy, similarity, and paired confidence intervals from the saved model outputs. Generated comparisons and audit files go under `derived/`, which is not tracked. Redundant chat-format datasets are also generated locally rather than stored again in Git.

## Train or generate new outputs

[sft.py](sft.py) trains LoRA adapters with loss on the assistant response. [eval.py](eval.py) evaluates the original model or a released adapter. [run_evals.sh](run_evals.sh) runs all four conditions on both evaluation splits. These steps need the model dependencies and suitable compute; they are not needed to read or recalculate our results. Follow the [evaluation instructions](docs/EVALUATION.md) before rerunning saved experiments.

[sonnet_rewrite.py](sonnet_rewrite.py) is the original cached rewrite runner. Its [documentation](docs/SONNET_RUNNER.md) explains the generation process. The accepted rewrites are already saved, so there is no need to pay to generate them again.

## Bottom line

SFT produced a stronger but inconsistent Mario voice and lower math accuracy. DPO has not shown a clear additional benefit in the completed automatic checks; persona-rubric judging remains to be done.

GSM8K: Cobbe et al. (2021), [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). The source revision and hashes are recorded in provenance; the upstream license is in [data/raw/LICENSE](data/raw/LICENSE). This is an unofficial educational Mario experiment, not endorsed by Nintendo.
