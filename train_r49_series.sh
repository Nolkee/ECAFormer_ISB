#!/bin/bash
# R49 Series: green-tint root fixes + crash elimination (built on R48b champion)
#
# DIAGNOSIS (2026-07-10, verified at code level):
# Early green = out ≈ mapping(≈0) + 0.6·x1 ≈ 0.9·x_low — a copy of the
# green-biased low-light input — made worse by EMA cold-start (validation uses
# net_g_ema; at iter 500 it is still ~61% random init).
# Mid-training PSNR crash (7-9.5K, SSIM/LPIPS unaffected) = global drift:
# nothing anchors illu_map scale, so the bridge endpoint x1 moves under the
# denoiser.
#
# R49a: R48b + residual_gray_world (G2) + ema_warmup (G1)  -> green gone
# R49b: R49a + anchor_loss 0.05 (S1)                       -> crash gone
# R49c: R49b + zero_init_mapping_bias (R41b perceptual win)-> SSIM/LPIPS push
#
# EXPECTED:
# - iter-500 baseline images color-neutral (NOT green)
# - no 7-9.5K PSNR valley
# - R49b/c: PSNR 22.4+, SSIM >= 0.80, LPIPS <= 0.16

set -e

CONFIGS=(
  "Options/ISB_ecaformer_r49a_gray_world_ema_warmup.yml"
  "Options/ISB_ecaformer_r49b_anchor_loss.yml"
  "Options/ISB_ecaformer_r49c_zero_bias_full.yml"
)

echo "=========================================="
echo "R49 series: green root fix + stability"
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
echo "R49 series completed."
echo "Check: visualization/baseline/ images should be color-neutral,"
echo "       no PSNR valley at 7-9.5K, best PSNR expected 22.4+."
echo "=========================================="
