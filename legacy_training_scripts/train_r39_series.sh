#!/bin/bash
# R39 Series: sigmoid activation + channel_scale green fix
#
# Root cause: identity+clamp blocks green channel gradients at boundaries
# Fix: sigmoid output activation (non-zero gradient everywhere) + disable clamps
# Combined with channel_scale [1.0, 0.95, 1.0] for source-level green suppression

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r39a_sigmoid_chscale.yml"
  "Options/ISB_ecaformer_r39b_sigmoid_base.yml"
  "Options/ISB_ecaformer_r39c_sigmoid_chscale_chroma.yml"
)

echo "=========================================="
echo "R39 series: sigmoid + channel_scale green fix"
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
echo "R39 series completed."
echo "=========================================="
