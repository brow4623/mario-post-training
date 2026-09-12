#!/usr/bin/env bash
# Run from the repository root after downloading the released adapters.
set -euo pipefail
results_dir=${1:-results/new-run}
provenance_dir=${2:-provenance/evaluation/new-run}
derived_dir=${3:-derived/new-run}
python prepare_data.py
for split in val test; do
  for condition in base prompted control mario; do
    args=()
    case "$condition" in
      prompted) args+=(--system-prompt-file mario_prompt.txt) ;;
      control|mario) args+=(--adapter "adapters/$condition") ;;
    esac
    python -u eval.py --data "data/$split.jsonl" --name "${condition}_${split}" \
      --batch-size 16 --max-new-tokens 1024 --results-dir "$results_dir" \
      --provenance-dir "$provenance_dir" --metrics-dir "$derived_dir/metrics" "${args[@]}"
  done
  python compare.py --split "$split" --results-dir "$results_dir" \
    --provenance-dir "$provenance_dir" --output-dir "$derived_dir/comparisons"
done
