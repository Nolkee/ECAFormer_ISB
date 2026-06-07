#!/bin/bash
# R38 Series: channel_scale green suppression
#
# Base: R37d (best PSNR, res06) + R36a (base)
# Goal: learnable per-channel illumination scale to suppress green tint at source
# Mechanism: channel_scale_init=[1.0, 0.95, 1.0] initializes green channel lower

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r38a_chscale.yml"
  "Options/ISB_ecaformer_r38b_base_chscale.yml"
  "Options/ISB_ecaformer_r38c_chscale_chroma.yml"
)

echo "=========================================="
echo "R38 series: channel_scale green suppression"
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
echo "R38 series completed."
echo "=========================================="
