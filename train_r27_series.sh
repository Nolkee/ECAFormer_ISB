#!/bin/bash
# R27 Series Ablation: r26b architecture + sigma_max=0 variants
# Short run: 12K iters, batch=24, patch=128, Adam lr=6e-5
#
# r27a: sigma_max=0 (no bridge noise)
# r27b: sigma_max=0 + decouple_x1_from_bridge
# r27c: sigma_max=0 + nfe=1 (single-step)
#
# Compare against r26b at same 12K: PSNR=20.69, SSIM=0.761, LPIPS=0.244

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r27a_sigma0.yml"
    "Options/ISB_ecaformer_r27b_sigma0_decouple.yml"
    "Options/ISB_ecaformer_r27c_sigma0_nfe1.yml"
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
