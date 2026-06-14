#!/bin/bash
# R45 Series: Source-level green correction (green_norm)
#
# ROOT CAUSE: x_low itself has green bias (Bayer sensor)
#             → x1 = x_low * (1 + illu_map) inherits it
#             → residual shortcut injects green bias into output
#
# SOLUTION: Fix x_low BEFORE x1 construction
#           green_norm: x_low[:, G] -= (g_mean - r_mean)
#
# R45a: green_norm only (cleanest test)
# R45b: green_norm + channel_scale (double correction)
# R45c: green_norm + green_loss (source + gradient guidance)
#
# HYPOTHESIS: R45a should match/beat R38c (22.10 PSNR) with NO green tint

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r45a_green_norm_only.yml"
  "Options/ISB_ecaformer_r45b_green_norm_plus_chscale.yml"
  "Options/ISB_ecaformer_r45c_green_norm_adaptive.yml"
)

echo "=========================================="
echo "R45 series: Source-level green correction"
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
echo "R45 series completed."
echo "=========================================="
