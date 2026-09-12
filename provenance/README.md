# Provenance

This directory contains the experiment records. The findings are in [the report](../reports/SFT_RESULTS.md); these files are for checking how the data and results were produced.

| Location | Contents |
| --- | --- |
| `data_manifest.json` | GSM8K source revision, split indices, and expected dataset hashes. |
| `generation/astra.jsonl`, `generation/sonnet.jsonl` | Accepted teacher-model answers, including the saved model/request metadata. These cannot be recovered by rerunning a deterministic script. |
| `generation/config.json` | Exact Sonnet prompt and generation settings. Kept unchanged, including historical filename references in the prompt. |
| `generation/attempts.jsonl` | API usage and rejected candidates from dataset generation. |
| `generation/caveats.jsonl`, [data review](data_review.md) | Source-answer concerns, sampled style review, and validation-example exposure. |
| `evaluation/*.config.json` | Full inference prompts, model revision, package versions, and code/data/adapter hashes for the eight completed runs. |
| `evaluation/run.log` | Original evaluation execution log. |

The recorded evaluation code hashes refer to commit `8602be9`, used for the September 12 runs. Later cleanup changed output locations, not the saved answers or scores. Do not replace the historical hashes with hashes of newer code. The saved inference configurations intentionally prevent resuming an old run with changed code; use a new output location for a new run.

Model outputs stay in [results](../results/). Dataset audits, intermediate chat records, generation summaries, per-run metrics, and comparison JSON are deterministic derivatives and are not tracked. Run `python prepare_data.py`, then `python compare.py --split val` and `python compare.py --split test` to rebuild them. Preparation writes audits and rewrite exports under `derived/`, plus ignored `data/original/` and `data/mario/` working files. Final training/evaluation datasets remain checked in for direct use.
