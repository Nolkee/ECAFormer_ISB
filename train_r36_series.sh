#!/bin/bash
# R36 Series: R33a-centered non-repeated ablation
#
# Base: R33a (illumination_channels=1)
# Goal: explore non-repeated tuning axes only

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r36a_nfeat80.yml"
  "Options/ISB_ecaformer_r36b_tv005.yml"
  "Options/ISB_ecaformer_r36c_color015.yml"
  "Options/ISB_ecaformer_r36d_mixup10.yml"
)

echo "=========================================="
echo "R36 series: R33a-centered non-repeated ablation"
echo "=========================================="

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "------------------------------------------"
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "------------------------------------------"

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

echo "=========================================="
echo "R36 series completed."
echo "=========================================="
