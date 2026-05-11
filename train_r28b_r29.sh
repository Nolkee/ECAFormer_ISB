#!/bin/bash
# One-click: R28b + R29 Phase II series
# Baseline: R28a PSNR=21.01, SSIM=0.797, LPIPS=0.164
#
# R28b: R28a + inference_steps=12
# R29a: R28a + ECA γ=1.5
# R29b: R28a + FFT Loss (weight=0.1)
# R29c: Full Phase II (γ=1.5 + FFT + eta_min=5e-7)

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r28b_eca_inference12.yml"
    "Options/ISB_ecaformer_r29a_eca_gamma15.yml"
    "Options/ISB_ecaformer_r29b_fft_loss.yml"
    "Options/ISB_ecaformer_r29c_phase2_full.yml"
)

echo "============================================"
echo "Training R28b + R29 series"
echo "Start: $(date)"
echo "============================================"

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yml)
    echo ""
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] Starting: $name"
    echo "=========================================="

    python -m basicsr.train --opt "$cfg" 2>&1 | tee "train_${name}.log"

    echo "[$(date +%H:%M:%S)] Finished: $name"
done

echo ""
echo "============================================"
echo "All experiments completed: $(date)"
echo "============================================"
