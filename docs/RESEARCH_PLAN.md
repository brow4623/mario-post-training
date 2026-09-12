# GSM8K Mario SFT research plan

This is the original proposed plan, not a completion tracker. The executed SFT-only protocol and its deviations are documented in `EVALUATION.md`; completed results are in `reports/SFT_RESULTS.md`. Pending entries below describe the original plan rather than the current submission status.

## Scope and hypotheses

Use Qwen/Qwen2.5-3B-Instruct, Mario, and exactly 3,000 training and 500 validation examples. This phase prepares the data and verifier; training, model generations, and hypothesis results are subsequent work, not completed deliverables.

- H0 (user's label): persona via prompting is weaker than persona via SFT.
- H1: persona training drops mathematical accuracy.

These are two directional research hypotheses, not a conventional null/alternative pair. For statistical testing, the corresponding nulls are no improvement in persona score and no decrease in math accuracy. Report effect sizes and uncertainty, including inconclusive results, rather than assuming either claim is true.

## Specific steps

**Revision after style review:** Steps 3-5 below describe the rejected first attempt and are superseded for persona generation by `prompts/mario_rewrite.md` and the approved examples in `prompts/mario_examples.md`. The replacement uses saved Astra rewrites and Claude Sonnet 5 API rewrites; see `SONNET_RUNNER.md` for resumable generation and publication. Keep the original numerical answers and ordered calculator annotations, but do not require exact recovery of the original prose. Human/semantic review is a separate gate because those automated checks cannot establish faithful reasoning or character quality. The 3,000/500 split and experimental comparisons remain unchanged. Do not use the historical insertion-based transformation for training.

1. Download the original GSM8K train/test JSONL and license from an immutable commit of the official repository. Record URLs, commit, SHA-256 hashes, seed, and source indices. Retain raw files unchanged.
2. Shuffle official training indices with seed 42. Assign 3,000 examples to train and the next 500 to validation. Leave remaining training examples unused. Reserve all 1,319 official test questions for final evaluation. Check IDs and exact question overlap across splits; note that exact checks do not establish absence of semantic duplicates or base-model pretraining contamination.
3. Define Mario's voice: upbeat, brief, encouraging, with restrained phrases such as "Let's-a go!" and "Wahoo!". Keep questions, entities, units, quantities, calculations, and final answers unchanged. Do not substitute game objects into math problems. Use the same neutral system instruction in the training data and unprompted evaluations so the SFT condition must learn the style from assistant targets.
4. Generate paired original and Mario JSONL records with stable source IDs and chat messages. Use deterministic, varied interjections before existing solution steps. This initial rule-based transformation preserves the full original solution, but has limited stylistic richness and may teach superficial catchphrases. Do not describe it as an LLM paraphrase.
5. Run the integrity checker over both splits. Require exact question/ID alignment, identical extracted final answers, identical ordered calculator annotations, and exact recovery of the original rationale after removal of the inserted style phrases. Inspect a seeded sample of at least 50 train and 50 validation pairs manually for readability, repetition, character voice, and semantic fidelity. Revise transformation rules and rerun all checks before training if the style is inadequate. Automated preservation does not establish that every source rationale is mathematically correct.
6. Freeze the dataset version and run baseline generations on the same validation questions. Compare four conditions: base + neutral instruction, base + Mario instruction, Mario-SFT + neutral instruction, and Mario-SFT + Mario instruction. Add a plain-GSM8K SFT control with the same 3,000 examples and training budget to separate effects of persona from generic math SFT.
7. Train an initial LoRA adapter on Qwen2.5-3B-Instruct. Start with rank 16, alpha 32, dropout 0.05, all linear layers, learning rate 2e-4, effective batch size 32, up to 3 epochs, and seed 42. These are proposed starting settings, not validated hyperparameters. Choose BF16 LoRA or 4-bit QLoRA after measuring GPU capacity. Use the model's chat template, assistant-only loss, and a sequence limit that includes the final answer; measure token lengths and never silently truncate targets. Save configuration, dependency versions, base revision, tokenizer, adapter, and training logs. Use validation only for selection; use a predeclared persona-score objective subject to an accuracy floor within 2 percentage points of the base validation score. If no checkpoint qualifies, report that rather than changing the threshold retrospectively.
8. Generate outputs for every evaluation condition with identical questions, neutral formatting instruction, greedy decoding, and the same token budget (initially 1,024 new tokens; check truncation on validation). Save only generated completion text, not the input prompt, as one `{id, output}` record per question. Record model/adapter revision, prompt, seed, decoding settings, and truncation status. Missing outputs count as wrong; report format failures separately. Never fill missing predictions with reference answers.
9. Evaluate math with exact normalized numeric match after a final `####` delimiter. Require this format in all prompts. Report accuracy, valid-format rate, counts, and paired accuracy differences; inspect malformed outputs separately because format failure is not necessarily arithmetic failure. The verifier checks final answers, not correctness of intermediate reasoning. Do not use an unsafe expression evaluator.
10. Evaluate persona with a blinded judge using a fixed 1-5 rubric: recognizable voice, consistent voice across the response, naturalness, and restraint. Hide condition labels and gold solutions, randomize output order, freeze judge model/prompt/settings, and have humans independently rate a fixed random sample of 100 outputs across conditions. Report agreement and examples of catchphrase gaming. Check accuracy separately rather than rewarding wrong but enthusiastic answers.
11. For the checkpoint's numerical similarity deliverable, fit a character 3-5-gram TF-IDF vectorizer on Mario training answers only. Transform held-out Mario reference answers and the original-model and SFT completions, then report mean paired cosine similarity with bootstrap intervals. Include base-neutral and base-Mario baselines. Repeat after masking numbers and calculator annotations and report a catchphrase-removed sensitivity analysis. This measures textual resemblance, not persona quality by itself; shared math content and templated phrases can inflate it. Never fit the vectorizer on test completions. Keep held-out reference answers out of generation prompts.
12. Freeze checkpoint and prompts, then evaluate all conditions once on the official test set. Report paired bootstrap 95% intervals (10,000 resamples, seed 42) for persona and accuracy differences and an exact McNemar test for paired correctness. Primary persona contrast: Mario-SFT neutral versus base Mario-prompted. Primary accuracy contrast: Mario-SFT neutral versus base neutral. Also compare Mario-SFT to plain SFT to assess persona-specific effects. Treat judge scores as noisy measurements and distinguish exploratory comparisons from primary contrasts. If resources permit, repeat training with three seeds; otherwise disclose single-run uncertainty.
13. Package the checkpoint submission: repository with runnable training/inference/evaluation code, reproducible environment and data manifest; trained adapter plus exact base revision or merged model with an accessible download and reload check; raw held-out generations; reference-to-output similarity scores for base and SFT; accuracy results and uncertainty; short examples and a limitations section. Do not commit large model binaries to normal Git. Later RLAIF/RLVR stages must reuse the frozen evaluation protocol and keep test data out of reward training.

## Planned result table

| Condition | Math accuracy | Format success | Persona judge | Reference cosine |
| --- | --- | --- | --- | --- |
| Base neutral | Pending | Pending | Pending | Pending |
| Base Mario prompt | Pending | Pending | Pending | Pending |
| Mario SFT neutral | Pending | Pending | Pending | Pending |
| Mario SFT + Mario prompt | Pending | Pending | Pending | Pending |
| Plain math SFT neutral | Pending | Pending | Pending | Pending |

Dataset answer-preservation rates are data QA, not entries in this model performance table.

## Sources

- Local assignment: `assignment #1 cs2881R .docx`, supplied by the user; establishes Qwen2.5-3B-Instruct and staged SFT/RLAIF/RLVR evaluation. The current requested execution is limited to planning, data preparation, and checking.
- [Official GSM8K repository](https://github.com/openai/grade-school-math): source format, calculator annotations, final-answer delimiter, and license.
- [GSM8K paper](https://arxiv.org/abs/2110.14168): Cobbe et al., Training Verifiers to Solve Math Word Problems.
