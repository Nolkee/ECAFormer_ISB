#!/bin/bash
# R47 Series: Return to original ECAFormer design (illumination_channels=3)
#
# ROOT CAUSE OF COLOR SHIFT:
# Current ISB implementation uses illumination_channels=1 (single-channel illumination)
# then manually expands to RGB with channel_scale=[1.0, 0.95, 1.0].
# This FORCES all RGB channels to share the same illumination estimate,
# then applies a fixed 5% green suppression — network cannot learn per-channel differences.
#
# Original ECAFormer used illumination_channels=3 (per-channel illumination),
# allowing the network to learn independent light enhancement for R/G/B.
# This is why original ECAFormer had NO color issues.
#
# R47a: Pure 3ch illumination (cleanest test, no manual corrections)
# R47b: 3ch illumination + green_norm (double insurance)
# R47c: 3ch illumination + green_norm + learnable per-channel residual (full control)
#
# HYPOTHESIS: R47a should match R38c (22.10) with natural colors.
#             R47b/c should exceed 22.2+ PSNR with perfect color balance.

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r47a_illum3ch_pure.yml"
  "Options/ISB_ecaformer_r47b_illum3ch_green_norm.yml"
  "Options/ISB_ecaformer_r47c_illum3ch_full_control.yml"
)

echo "=========================================="
echo "R47 series: Per-channel illumination"
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
echo "R47 series completed."
echo "=========================================="
