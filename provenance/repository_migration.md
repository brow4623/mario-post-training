# Repository migration

On September 18, 2026, we copied `brow4623/hw1-tantia-brown` to the empty course repository, `harvard-cs2881f26/brown-ogbogu-tantia`.

- Source main: `34a6e8f0b8ace2e7c7541f833aadb0b4dcfb9280`. All 26 commits reachable from that main were copied with their original IDs, parents, authors, messages, and timestamps. No history was squashed or rewritten.
- The original `adapters-v1` tag still points to `dbea375dae6b687068192985c006a9e192ac017d`.
- The old pull request's head is already an ancestor of main. Its separate GitHub-generated merge commit is preserved under `legacy-pr-1-merge` (`aac01f2ff7dfdeebb72215c865cc91a5bd80ee94`).
- Both release weight files were copied unchanged. Their SHA-256 values match `adapters/README.md` and the destination release asset digests. The new release has a new publication date; the weights and Git tag are unchanged.
- GitHub pull-request discussions and other hosting metadata were not migrated. The original repository remains available and unchanged.

The new local checkout is `C:/Users/kanis/Documents/Codex/brown-ogbogu-tantia`. Its `origin` points to the course repository; `legacy` points to the source repository.
