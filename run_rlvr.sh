#!/usr/bin/env bash
# Stage 3 RLVR: pinned environment, offline tests, smoke test, pilot, two training runs, and
# the matched evaluation of all three conditions. Run from the repository root on the pinned
# PyTorch 2.8 / CUDA 12.8 image, with adapters/rlaif/adapter_model.safetensors downloaded
# (see adapters/README.md). The combined run needs ANTHROPIC_API_KEY in an ignored .env.
#
# Scoring the evaluation outputs with the persona judge is a separate, GPU-free step; see
# docs/RLVR.md. Check the RunPod and Anthropic balances before starting: this script rents
# the GPU for both runs and spends the judge budget below.
set -euo pipefail
results_dir=${1:-results/rlvr}
provenance_dir=${2:-provenance/evaluation/rlvr}
judge_budget=${RLVR_JUDGE_BUDGET_USD:-6}
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p provenance/rlvr
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

# 4. Matched evaluation: one greedy answer per question, identical settings for all three
#    conditions, including the re-evaluated RLAIF baseline.
for condition in rlvr_baseline rlvr_math rlvr_combined; do
  case "$condition" in
    rlvr_baseline) adapter=adapters/rlaif ;;
    rlvr_math) adapter=checkpoints/rlvr-math ;;
    rlvr_combined) adapter=checkpoints/rlvr-combined ;;
  esac
  for split in val test; do
    python -u eval.py --data "data/${split}.jsonl" --name "${condition}_${split}" \
      --adapter "$adapter" --batch-size 16 \
      --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
      --results-dir "$results_dir" --provenance-dir "$provenance_dir" \
      --metrics-dir derived/rlvr/metrics
  done
done
echo "Download checkpoints/rlvr-math, checkpoints/rlvr-combined, ${results_dir}, ${provenance_dir},"
echo "and provenance/rlvr before releasing the pod. Then score the persona ratings per docs/RLVR.md."
