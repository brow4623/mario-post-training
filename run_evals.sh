#!/usr/bin/env bash
# Run from the repository root after downloading the released adapters.
set -euo pipefail
for split in val test; do
  for condition in base prompted control mario; do
    args=()
    case "$condition" in
      prompted) args+=(--system-prompt-file mario_prompt.txt) ;;
      control|mario) args+=(--adapter "adapters/$condition") ;;
    esac
    python -u eval.py --data "data/$split.jsonl" --name "${condition}_${split}" \
      --batch-size 16 --max-new-tokens 1024 "${args[@]}"
  done
  python compare.py --split "$split"
done
