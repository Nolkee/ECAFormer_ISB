#!/bin/bash
# R46 Series: R45 fix — green_norm in inference
#
# R45 FAILURE ROOT CAUSE:
# green_norm only applied during training → train/val data distribution mismatch
# Training: model saw balanced x_low (green corrected)
# Validation: model saw raw x_low (green bias) → 泛化失败
#
# R46 FIX:
# Remove `self.training` check → green_norm applies in inference too
# Now train/val see the same data distribution
#
# R46a: pure green_norm (cleanest test)
# R46b: green_norm + per-ch residual_scale [0.6, 0.55, 0.6] (milder than R42a)
#
# HYPOTHESIS: R46a should match/beat R38c (22.10) with NO green tint

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r46a_green_norm_inference.yml"
  "Options/ISB_ecaformer_r46b_green_norm_plus_per_ch_res.yml"
)

echo "=========================================="
echo "R46 series: green_norm inference fix"
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
echo "R46 series completed."
echo "=========================================="
