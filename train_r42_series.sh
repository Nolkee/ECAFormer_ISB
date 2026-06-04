#!/bin/bash
# R42 Series: 4 orthogonal green tint fixes (A/B/C/D) + combined (E)
#
# Root cause: x1 = x_low * (1 + illu_map) carries x_low's green bias
# through residual shortcut. R42 tests 4 independent fixes:
#   R42a: per-channel residual_scale=[0.6, 0.5, 0.6] — less green residual
#   R42b: green_norm=true — equalize green/red input mean
#   R42c: green_loss_weight=0.1 — penalize green excess in output
#   R42d: channel_permute_prob=0.3 — random RGB permutation augmentation
#   R42e: all 4 combined

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r42a_per_ch_res.yml"
  "Options/ISB_ecaformer_r42b_green_norm.yml"
  "Options/ISB_ecaformer_r42c_green_loss.yml"
  "Options/ISB_ecaformer_r42d_ch_permute.yml"
  "Options/ISB_ecaformer_r42e_all_green_fixes.yml"
)

echo "=========================================="
echo "R42 series: 4 orthogonal green tint fixes"
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
echo "R42 series completed."
echo "=========================================="
