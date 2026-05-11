#!/bin/bash
# R29 Series: Phase II optimization ablation
# Base: R28a (embed_dim=64, ECA, AdamW wd=0.05)
#
# r29a: ECA γ=1.5 (broader channel attention)
# r29b: + FFT Loss (weight=0.1, magnitude+phase)
# r29c: Full Phase II (γ=1.5 + FFT + eta_min=5e-7 + T=12)
#
# Compare against R28a: PSNR=21.01, SSIM=0.797, LPIPS=0.164

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r29a_eca_gamma15.yml"
    "Options/ISB_ecaformer_r29b_fft_loss.yml"
    "Options/ISB_ecaformer_r29c_phase2_full.yml"
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

echo "R29 ablation completed."
