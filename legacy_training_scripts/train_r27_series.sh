#!/bin/bash
# R27 Series Ablation: t-clamp fix + r26b variants
# Bug fix: train t∈[0.01,1.0] instead of [0.01,0.99] to match inference t=1.0
# Short run: 12K iters, batch=24, patch=128, Adam lr=6e-5
#
# r27a: t-clamp fix only
# r27b: t-clamp fix + decouple_x1_from_bridge
# r27c: t-clamp fix + nfe=1 (single-step)
#
# Compare against r26b at same 12K: PSNR=20.69, SSIM=0.761, LPIPS=0.244

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r27a_tfix.yml"
    "Options/ISB_ecaformer_r27b_tfix_decouple.yml"
    "Options/ISB_ecaformer_r27c_tfix_nfe1.yml"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "=========================================="

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

echo "R27 ablation completed."
