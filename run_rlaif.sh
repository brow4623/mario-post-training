#!/usr/bin/env bash
# Run from the repository root on the pinned PyTorch 2.8 / CUDA 12.8 image.
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p provenance/training/rlaif
# The image includes nightly vision/audio extensions incompatible with stable torch.
# This text-only experiment needs neither extension.
pip uninstall -y torchvision torchaudio
pip install -r requirements-rlaif.txt
pip freeze > provenance/training/rlaif/packages.txt
python -m unittest discover -s tests -q
python -u dpo.py --prefs data/prefs.jsonl --adapter adapters/mario \
  --output checkpoints/rlaif-smoke --max-examples 8 --max-steps 2 \
  --batch-size 2 --grad-accum 1 --warmup-steps 0
python -u eval.py --data data/val.jsonl --name rlaif_smoke \
  --adapter checkpoints/rlaif-smoke --max-examples 8 --batch-size 8 \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --results-dir derived/smoke --provenance-dir derived/smoke --metrics-dir derived/smoke
python -u dpo.py --prefs data/prefs.jsonl --adapter adapters/mario --output checkpoints/rlaif
for condition in rlaif mario_rlaif_baseline; do
  adapter=checkpoints/rlaif
  if [[ "$condition" == mario_rlaif_baseline ]]; then adapter=adapters/mario; fi
  for split in val test; do
    python -u eval.py --data "data/${split}.jsonl" --name "${condition}_${split}" \
      --adapter "$adapter" --batch-size 16 \
      --revision aa8e72537993ba99e69dfaafa59ed015b17504d1
  done
done
