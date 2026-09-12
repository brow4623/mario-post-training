# Data preparation verification

These results describe dataset integrity, not trained-model performance.

For the completed model evaluations, numerical similarity comparisons, and hypothesis conclusions, see [SFT_RESULTS.md](SFT_RESULTS.md). The tables below remain dataset-preparation QA only.

The complete context-aware replacement contains 475 preserved Astra rewrites and 3,025 Claude Sonnet 5 rewrites (low effort). It replaces the rejected insertion-based dataset. Passing numerical checks do not establish persona quality or semantic fidelity.

| Check | Result |
| --- | --- |
| Train original/Mario integrity | 3,000 / 3,000 passed |
| Validation original/Mario integrity | 500 / 500 passed |
| Original training final answers against themselves | 3,000 / 3,000 matched |
| Mario training final answers against original | 3,000 / 3,000 matched |
| Mario validation final answers against original | 500 / 500 matched |
| Official test reference format validation | 1,319 / 1,319 valid |
| Unit and downloaded-data tests | 28 passed |

Rewrite integrity checks include unchanged questions, identical ordered calculator annotations, exact final lines, normalized final answers, and chat-message consistency. Original prose recovery is deliberately not required for genuine rewrites. Tests also check stored SHA-256 hashes, reconstruction from raw source records and rewrite shards, exact split disjointness, malformed outputs, missing predictions, duplicate IDs, deliberately corrupted answers, and runner caching/retry behavior.

The explanations were rewritten in contextual Mario voice, with source questions and numerical targets preserved. Spot-checks and limitations are recorded in `mario_v2_review.md`; the planned 50-train/50-validation human style review has not been completed. Original-source ambiguities and occasional rewrite embellishments remain review caveats.

Sonnet generation received 3,064 API responses, rejected 39 candidates before saving, and finished with zero missing records. Cache reads accounted for 91.88% of reported input tokens; estimated API cost was $11.41, including rejected attempts. This is a usage-based estimate, not a billing invoice. Exact usage and provenance are in `data/generation/summary.json` and the compact raw records in `data/generation/sonnet.jsonl`.

The official test reference self-check below only validates parsing, not model performance or independent correctness of the source solutions. Model performance is reported separately in `SFT_RESULTS.md`; no independent persona judge score was collected.

Each JSON summary has a corresponding `.details.jsonl` with per-example results. Reproduction commands are in the root README.
