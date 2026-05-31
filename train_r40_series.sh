#!/bin/bash
# R40 Series: stronger green suppression
#
# Base: R38c (best PSNR 22.10, res06 + chscale + chroma)
# Problem: R38c still has early green tint — channel_scale 0.95 is too weak
# Solution: stronger channel_scale init + lower residual_scale + stronger color loss
# R39 sigmoid failed (PSNR max 19, crashes) — abandoning sigmoid approach

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r40a_chscale090.yml"
  "Options/ISB_ecaformer_r40b_chscale085_res05.yml"
  "Options/ISB_ecaformer_r40c_chscale090_color02.yml"
)

echo "=========================================="
echo "R40 series: stronger green suppression"
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
echo "R40 series completed."
echo "=========================================="
