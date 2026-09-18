# RLAIF persona judge run

We rated 2,000 saved outputs: Mario SFT and RLAIF, each on 500 validation and 500 test questions. Both conditions used the same frozen rubric, individual-answer API protocol, temperature 0, and a maximum of 512 output tokens. The model alias resolved to `claude-haiku-4-5-20251001` for every rating.

The first 20 answers formed a transport, parsing, and cost pilot. We kept those scores and resumed the remaining 1,980 requests with four workers. We did not modify the rubric after reading pilot results. The full run used an $8 recorded-spend ceiling with conservative in-flight reservations. The pilot cost approximately $0.0413; the completed 2,000 calls cost approximately $4.0757 at the verified Haiku 4.5 standard rates. These are token-based estimates, not a billing invoice.

- `config.json` contains the full judge configuration and its hash. It is separate from the older Claude Code judge configuration.
- `attempts.jsonl` contains all 2,000 API responses' usage and request metadata. Every recorded attempt was accepted. No HTTP errors were recorded.
- The four score files in `results/judge/` retain each answer hash, configuration hash, resolved model, rating, and short explanation. `evaluate_persona.py summarize` checks coverage, hashes, and model consistency before comparing scores.
- `human_review_key.json` maps the 50 randomly selected question pairs to conditions. **Human reviewers should not open this file until they finish their ratings.** The public-facing review packet and blank rating sheet are under `reports/persona_review/`.

The API backend differs from the older Claude Code batch backend, so compare the two freshly scored conditions, not their absolute scores against the older batch ratings. The rubric and judge family were also used to create DPO training preferences; independent human review remains necessary.

Pricing used: $1 per million uncached input tokens, $1.25 per million 5-minute cache-write tokens, $0.10 per million cache-read tokens, and $5 per million output tokens. Source: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing). Regenerate detailed cost and statistical summaries with `python evaluate_persona.py summarize`; they are not duplicated here.
