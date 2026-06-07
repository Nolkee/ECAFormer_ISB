#!/bin/bash
# R37 Series: fix green tint + desaturation
#
# Base: R36a (n_feat=80, illumination_channels=1)
# Goal: reduce residual shortcut green tint propagation + add chroma loss for saturation
# Root causes identified from code analysis:
#   - Green: residual_scale * x1 where x1 = x_low * illu_map + x_low carries input tint
#   - Desaturation: color_loss only checks channel mean, no saturation penalty

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r37a_res05.yml"
  "Options/ISB_ecaformer_r37b_chroma.yml"
  "Options/ISB_ecaformer_r37c_res05_chroma.yml"
  "Options/ISB_ecaformer_r37d_res06.yml"
)

echo "=========================================="
echo "R37 series: fix green tint + desaturation"
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
echo "R37 series completed."
echo "=========================================="
