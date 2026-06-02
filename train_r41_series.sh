#!/bin/bash
# R41 Series: architectural green tint fixes
#
# Root cause: x1 = x_low * (1 + illu_map) carries x_low's green bias
# through residual shortcut. channel_scale operates on illumination (wrong target).
# R41 tests three orthogonal architectural fixes:
#   R41a: post GroupNorm(1,3) — normalize across RGB channels at output
#   R41b: zero mapping bias — eliminate random per-channel offset
#   R41c: 3ch illumination — estimator learns per-channel correction
#   R41d: post norm + 3ch illumination — combined fix

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r41a_post_norm.yml"
  "Options/ISB_ecaformer_r41b_zero_bias.yml"
  "Options/ISB_ecaformer_r41c_illum3ch.yml"
  "Options/ISB_ecaformer_r41d_post_norm_illum3ch.yml"
)

echo "=========================================="
echo "R41 series: architectural green tint fixes"
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
echo "R41 series completed."
echo "=========================================="
