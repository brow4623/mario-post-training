# Evaluate the RLAIF persona

The automated evaluation is complete: DPO scored higher under the frozen Mario rubric on both splits. See the [results](../reports/RLAIF_RESULTS.md). The [blinded human-review packet](../reports/persona_review/review.md) is ready, but human ratings are still pending.

We tested persona performance rather than reference wording. Both conditions had 500 validation and 500 test outputs, so judging needed no GPU or new model generation. The protocol below records how we evaluated them.

## Protocol

1. Use Haiku 4.5 with the unchanged `prompts/mario_judge.md`. The judge sees only each question and answer, not the model label, gold answer, or correctness. Score both conditions again with the same backend; do not combine new API scores with the older Claude Code scores.
2. Run a small pilot to check API access, response parsing, and actual usage. Do not revise the rubric in response to which model wins. Resume the same score files for the full run.
3. Score all 2,000 outputs. Report validation and test separately, with paired overall-score differences and 95% bootstrap intervals. Also report voice, consistency, naturalness, restraint, and the catchphrase-only rate. Do not select a new checkpoint using test results.
4. Check the verbosity concern: report answer lengths and the paired score difference among questions whose two answers are within 10% of each other in word count. Report the subset size. This is a sensitivity check, not proof that length causes the scores.
5. Prepare a blinded random sample of 50 question pairs, 25 from each split, using seed 42. Randomize A/B order and keep the condition key separately under provenance. A human should rate both answers with the same rubric before seeing model labels or automated scores. Focus on reactions to the problem, sustained voice, irrelevant lore, and unnecessary length. This gives 100 human-rated outputs; do not present an assistant's own review as independent human validation.

The primary outcome is the paired difference in the judge's overall persona score. A positive interval excluding zero supports improvement under this judge. An interval including zero leaves the result inconclusive, rather than proving equivalence. Agreement with the human review matters because DPO was trained using this same judge family.

## Access and commands

Put `ANTHROPIC_API_KEY` in the ignored root `.env`. This requires Anthropic API credit, not RunPod credit. Check pilot usage before committing to the full API bill. The commands below retain the frozen Haiku rubric but use the API backend consistently for both conditions.

To recalculate the completed evaluation for free, run `python evaluate_persona.py summarize`. Use the scoring commands only for new output files or an incomplete run. A limited pilot must not overwrite a completed score file; the runner refuses that operation.

```bash
# Pilot: 10 answers per condition. Scores are resumable.
python judge.py score results/mario_rlaif_baseline_val.jsonl results/rlaif_val.jsonl --env-file .env --limit 10 --workers 2 --max-tokens 512 --budget-usd 8 --provenance-dir provenance/judge/rlaif-api

# Full run: resumes the pilot, then scores both splits.
python judge.py score results/mario_rlaif_baseline_val.jsonl results/rlaif_val.jsonl results/mario_rlaif_baseline_test.jsonl results/rlaif_test.jsonl --env-file .env --workers 4 --max-tokens 512 --budget-usd 8 --provenance-dir provenance/judge/rlaif-api

python judge.py summarize results/judge/mario_rlaif_baseline_val.jsonl results/judge/rlaif_val.jsonl
python judge.py summarize results/judge/mario_rlaif_baseline_test.jsonl results/judge/rlaif_test.jsonl
```

Store scores in `results/judge/`, run settings and attempts in `provenance/judge/rlaif-api/`, and update `reports/RLAIF_RESULTS.md` with the findings. Keep regenerated statistical summaries under `derived/`.
