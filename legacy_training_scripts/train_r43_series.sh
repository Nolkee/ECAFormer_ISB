#!/bin/bash
# R43 Series: Root-cause fix for green tint via identity_scale in x1 construction
#
# Core innovation: x1 = x_low * illu_map + identity_scale * x_low
# Directly suppresses green bias in the x_low shortcut (2x weight injection point)
#
# R43a: identity_scale=[1.0, 0.92, 1.0] + neutral residual_scale=0.6
# R43b: identity_scale=[1.0, 0.90, 1.0] + per-channel residual_scale=[0.6, 0.5, 0.6]
# R43c: R43a + green_loss_weight=0.1 (triple defense)

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r43a_identity_scale.yml"
  "Options/ISB_ecaformer_r43b_double_suppress.yml"
  "Options/ISB_ecaformer_r43c_identity_green_loss.yml"
)

echo "=========================================="
echo "R43 series: Identity scale green fix"
echo "Root cause: x1 = x_low * illu_map + x_low carries green bias at 2x weight"
echo "Solution: x1 = x_low * illu_map + identity_scale * x_low with identity_scale=[1.0, 0.90-0.92, 1.0]"
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
echo "R43 series completed."
echo "Expected: PSNR 22.3-22.5, SSIM 0.795+, LPIPS 0.160-"
echo "Compare to R38c baseline: PSNR 22.10, SSIM 0.788, LPIPS 0.166"
echo "=========================================="
