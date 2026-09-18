# RLAIF training records

We trained the Mario SFT adapter on September 18, 2026, using 773 correct/correct preference pairs, one epoch, and 49 optimizer steps. The final adapter SHA-256 is `278224dcc00ea44511fcce4a760e6f50e5a99813dacd73baaae1837e6f158e58`.

- `run_manifest.json` records the input hashes, model revision, dependency versions, and reference-model hash before training. Commit `7def4b2` contains the exact executed `dpo.py` source.
- `checks.json` confirms that all 504 adapter tensors changed, all trainable parameters remained finite, and the reference-model hash was unchanged.
- `train_results.json` and `trainer_state.json` contain aggregate metrics and the step-by-step training log. `train_args.json` records the CLI arguments; `packages.txt` records the installed environment.
- `smoke/` contains the two-step training checks. The smoke adapter was never used to initialize the full run.
- `token_length_audit.json` corrects a preflight reporting bug: the original check counted the two fields of a tokenizer `BatchEncoding`, not token IDs. The corrected audit checked all 1,546 sequences; the longest was 445 tokens, below the configured 1,536. This bug did not truncate or change the training data. We retained the original manifest and fixed the check with a regression test.

The training image included incompatible nightly vision/audio packages. We removed them and installed the versions in `requirements-rlaif.txt` before the successful smoke and full runs. The model is text-only. We evaluated both the new adapter and the original Mario adapter under this same environment rather than comparing across library versions.
