#!/bin/bash
# R27 Series: r26b architecture with sigma_max=0 variants
# All configs: 250K iters, batch=8, patch=256, Adam lr=2e-4, Charbonnier+VGG
#
# r27a: sigma_max=0 (no bridge noise, deterministic interpolation)
# r27b: sigma_max=0 + decouple_x1_from_bridge (bridge from x_low, x1 as cond)
# r27c: sigma_max=0 + nfe=1 (single-step, no train-test NFE mismatch)
#
# Baselines for comparison:
#   ECAFormer paper: PSNR=24.24, SSIM=0.850
#   r26b (12K iters): PSNR=20.69, SSIM=0.761, LPIPS=0.244

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r27a_sigma0.yml"
    "Options/ISB_ecaformer_r27b_sigma0_decouple.yml"
    "Options/ISB_ecaformer_r27c_sigma0_nfe1.yml"
)

# Optional: resume from checkpoint by setting RESUME env var
# Example: RESUME=experiments/ISB_ecaformer_r27a_sigma0/train/iter_50000.pth bash train_r27_series.sh
RESUME="${RESUME:-}"

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo "=========================================="
    echo "Training: $name"
    echo "Config:   $cfg"
    echo "Time:     $(date)"
    echo "=========================================="

    extra_args=""
    if [ -n "$RESUME" ]; then
        # Find matching resume checkpoint for this config
        resume_path="${RESUME/ISB_ecaformer_r27a_sigma0/$name}"
        if [ -f "$resume_path" ]; then
            extra_args="--resume $resume_path"
            echo "Resuming from: $resume_path"
        fi
    fi

    python -m basicsr.train --opt "$cfg" $extra_args 2>&1 | tee "train_${name}.log"

    echo "Finished: $name at $(date)"
    echo ""
done

echo "All R27 experiments completed."
