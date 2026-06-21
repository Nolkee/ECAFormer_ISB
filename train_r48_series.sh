#!/bin/bash
# R48 Series: Root-cause fixes for illumination_channels=3 instability
#
# DIAGNOSIS FROM AGENT ANALYSIS (2026-06-22):
# R47 series (illumination_channels=3) all crashed at 6K-8K iterations.
# Root causes identified:
# 1. AdaLN zero-init (40%): Brittle learned corrections → gradient explosion on distribution shift
# 2. Bridge noise uniform (20%): Green channel gets larger gradients → amplifies bias
# 3. x1 construction (40%): + identity_scale * x_low doubles green bias from input
#
# R48a: AdaLN stabilization (non-zero init + extended warmup + stronger color_loss)
# R48b: Per-channel bridge noise weighting (reduce green noise by 20%)
# R48c: x1 construction fix (identity_scale=[1.0, 0.92, 1.0] + long warmup)
#
# HYPOTHESIS:
# - R48a: Most likely to succeed (stabilizes AdaLN learning)
# - R48b: Reduces green amplification at source (bridge loss)
# - R48c: Combines 3ch illumination + x1 green suppression (but may conflict)
#
# EXPECTED: All reach PSNR 22.0+, stable through 24K, natural colors.

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r48a_illum3ch_adaln_stable.yml"
  "Options/ISB_ecaformer_r48b_illum3ch_bridge_reweight.yml"
  "Options/ISB_ecaformer_r48c_illum3ch_x1_fix.yml"
)

echo "=========================================="
echo "R48 series: Root-cause fixes for 3ch illumination"
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
echo "R48 series completed."
echo "Expected: PSNR 22.0+, no instability, natural colors."
echo "=========================================="
