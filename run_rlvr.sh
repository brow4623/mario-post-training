#!/usr/bin/env bash
# Stage 3 RLVR: pinned environment, offline tests, smoke test, pilot, two training runs, and
# the matched evaluation of all three conditions. Run from the repository root on the pinned
# PyTorch 2.8 / CUDA 12.8 image, with adapters/rlaif/adapter_model.safetensors downloaded
# (see adapters/README.md). The combined run needs ANTHROPIC_API_KEY in an ignored .env.
#
#   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# torch 2.8.0 has since been withdrawn from PyPI, so it can no longer be installed: it has to
# come from the pod image, which is why that exact image matters. The check below fails fast
# rather than letting pip produce a confusing "no matching distribution" several minutes in.
#
# Optional persona-weight sweep (docs/RLVR_SENSITIVITY.md), off by default because it adds
# three more training runs and three more evaluation pairs to the matched two-condition design:
#   RLVR_SENSITIVITY_WEIGHTS="0.5 0.75 1.0" bash run_rlvr.sh
#
# Scoring the evaluation outputs with the persona judge is a separate, GPU-free step; see
# docs/RLVR.md. Check the RunPod and Anthropic balances before starting: this script rents
# the GPU for every run and spends the judge budget below on each combined one.
set -euo pipefail
results_dir=${1:-results/rlvr}
provenance_dir=${2:-provenance/evaluation/rlvr}
judge_budget=${RLVR_JUDGE_BUDGET_USD:-6}
# Per-run ceiling against the shared ledger, not a total across runs.
sensitivity_weights=${RLVR_SENSITIVITY_WEIGHTS:-}
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p provenance/rlvr

torch_version=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
case "$torch_version" in
  2.8.*) echo "torch $torch_version from the pod image" ;;
  *) echo "This pod has torch '$torch_version'; requirements-rlvr.txt pins 2.8.0, which PyPI no" >&2
     echo "longer serves. Start the pod on the image named at the top of this script." >&2
     exit 1 ;;
esac

# The image includes nightly vision/audio extensions incompatible with stable torch.
pip uninstall -y torchvision torchaudio
pip install -r requirements-rlvr.txt
pip freeze > provenance/rlvr/packages.txt
python -m unittest discover -s tests -q
python make_rlvr_data.py --check

# 1. Two-update smoke test on the real adapter: no judge calls, throwaway checkpoint. Read
#    provenance/rlvr/smoke/samples.jsonl and hand-check the rewards before going further.
python -u rlvr.py --reward math --adapter adapters/rlaif --output checkpoints/rlvr-smoke \
  --run-name smoke --max-examples 8 --max-steps 2 --batch-size 4 --grad-accum 1 \
  --max-completion-length 256

# 2. Pilot at the real sampling settings with the learning rate at zero, so the group
#    statistics describe the starting policy. Groups whose four answers are all correct or
#    all wrong teach GRPO nothing; too few mixed groups stops the script before it spends.
python -u rlvr.py --reward math --adapter adapters/rlaif --output checkpoints/rlvr-pilot \
  --run-name pilot --max-examples 64 --lr 0
python rewards.py summarize provenance/rlvr/pilot/rewards.jsonl --require-mixed-groups 10

# 3. The two training runs: independent, identical except for the reward. Never start the
#    combined run from the math-only adapter.
python -u rlvr.py --reward math --adapter adapters/rlaif --output checkpoints/rlvr-math
python -u rlvr.py --reward combined --adapter adapters/rlaif --output checkpoints/rlvr-combined \
  --env-file .env --budget-usd "$judge_budget"

conditions=("rlvr_baseline adapters/rlaif"
            "rlvr_math checkpoints/rlvr-math"
            "rlvr_combined checkpoints/rlvr-combined")

# 3b. Optional sweep: the same run at other persona weights, each from the RLAIF adapter
#     again. The judge is called for every correct completion whatever the weight, so these
#     cost roughly one combined run each; the weight only scales the reward afterwards.
for weight in $sensitivity_weights; do
  slug="w${weight//./_}"
  python -u rlvr.py --reward combined --adapter adapters/rlaif \
    --output "checkpoints/rlvr-combined-${slug}" --persona-weight "$weight" \
    --env-file .env --budget-usd "$judge_budget"
  conditions+=("rlvr_combined_${slug} checkpoints/rlvr-combined-${slug}")
done

# 4. Matched evaluation: one greedy answer per question, identical settings for every
#    condition, including the re-evaluated RLAIF baseline.
for entry in "${conditions[@]}"; do
  read -r condition adapter <<<"$entry"
  for split in val test; do
    python -u eval.py --data "data/${split}.jsonl" --name "${condition}_${split}" \
      --adapter "$adapter" --batch-size 16 \
      --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
      --results-dir "$results_dir" --provenance-dir "$provenance_dir" \
      --metrics-dir derived/rlvr/metrics
  done
done
echo "Download checkpoints/, ${results_dir}, ${provenance_dir}, and provenance/rlvr before"
echo "releasing the pod. Then score the persona ratings per docs/RLVR.md."
