# Context-aware rewrite review

## Acceptance criteria

The original insertion-based dataset was rejected by the user. The replacement must rewrite the explanatory prose itself in Mario's voice, with reactions grounded in each question and characterful narration of the calculations. A neutral first-person paraphrase is also insufficient. Neither catchphrase counts nor numerical preservation establish persona quality.

Numeric final-answer equality, exact ordered calculator annotations, stable IDs, and unchanged questions/chat pairing are checked automatically. These do not prove semantic fidelity of the prose. The reviewing assistant and generation subagents must also check relationships, units, unannotated steps, and contextual voice. Human review by the project team remains advisable before training.

## Calibration

Early calibration used `gpt-5.6-luna` subagents. Those attempts passed numeric checks but failed the voice review and are excluded from the final dataset. The final data uses Astra and Sonnet, not Luna.

The reviewing assistant compared source questions, source solutions, and rewrites at train indices 0, 1, 4, 9, 17, 25, 32, 40, and 49. Corrections included distinguishing Tiffany's relay-leg duration from her finish order and retaining the fact that Allie is included in the party's 42 people. These are examples of errors that final-answer/annotation checks alone do not detect.

The latest samples use contextual dialogue about dinner, plumbing, charging devices, a relay team, party food, silk pillowcases, chocolate shopping, and stamp collections. Sensitive illness examples are intentionally less exuberant. This calibration review is not a claim that every record in the full replacement has been independently reviewed.

## Astra review

After Luna failed the style gate, the user approved Astra with medium reasoning. Three Astra workers produced 475 saved rewrites (325 train, 150 validation) before the user switched to Sonnet. Those answers are preserved in `provenance/generation/astra.jsonl`; partial checkpoints and redundant batch files were removed during repository cleanup. Training workers inadvertently saw three validation examples in the shared style document and were instructed not to use them as training examples. This exposure is a limitation of the Astra subset. Sonnet received only four training examples. The official test set was not rewritten or used as generation examples.

The parent sampled Astra train indices 0, 4, 17, 25, 29, 32, 40, 49, 1504, 1516, 1533, and 1549, and validation indices 4, 16, 33, and 49. These samples were more context-aware than the rejected Luna outputs. This does not establish uniform quality across all saved examples. Source caveats and obvious unannotated prose/algebra clarifications are recorded separately by each worker under `provenance/generation/caveats.jsonl`.

## Sonnet review

Generation completed on 2026-09-10: 3,025 Sonnet answers plus the unchanged 475 Astra answers were published as 3,000 training and 500 validation records. Both complete splits passed all deterministic integrity checks. No incomplete split was published.

The user subsequently requested Claude Sonnet 5 with low effort for the remaining examples. The cached prefix uses four approved training examples, not validation examples. The runner preserves the 475 Astra answers, checkpoints each accepted Sonnet answer, and regenerates rejected candidates rather than manually repairing their math. Full usage, rejected attempts, and model provenance are retained under `provenance/generation`.

The reviewing assistant compared source questions, solutions, and Sonnet rewrites for training IDs 1784, 394, 4635, 5358, 5851, 6474 (pilot), and 5557, 1743, 1186, 6251, 2932, 2703, 2528, 1930. Validation samples were IDs 3315, 1375, and 5351. All IDs have prefix `gsm8k/train/` because both splits derive from official training data. The latter training and validation samples used seed 42 over available saved files at review time, not a prespecified full-dataset sample.

The sampled voice reacts to the actual food, animals, vehicles, purchases, and operations rather than inserting unrelated catchphrases. However, occasional flourishes add unsupported details, and validation ID 5351 calls the final 13 carts "13 trips" in its closing line. Its calculation and gold answer remain correct, but that unit wording is a semantic-review caveat. These samples do not establish uniform persona quality or semantic fidelity. The planned 50-train/50-validation human review has not been completed; numerical checks are not a substitute. Original source caveats remain in `provenance/generation/caveats.jsonl`.
