#!/bin/bash
# R51 Series: anchor-strength sweep + deadzone anchor (on the r50a base:
# gray-world decay 1500-3500, ema_warmup, NO estimator_lr_mult).
#
# R50 VERDICT (2026-07-15):
# - r50a (no anchor): matched champion (PSNR 22.21 / SSIM 0.7964 / LPIPS
#   0.1603 — first config to beat R48b on LPIPS without losing PSNR/SSIM),
#   desaturation FIXED, but the 6-7K crash remains (-2.4 dB valley).
# - r50b (estimator_lr_mult 0.3): REFUTED — slowing the estimator made the
#   transient longer and deeper (peak 20.83, early-stopped ~8.5K). The drift
#   needs a restoring force, not a slower clock.
# - r50c (anchor x1_lq w0.5): crash ELIMINATED (first no-valley curve in
#   project history; SSIM 0.8064 record) but PSNR capped at 21.75 — exact
#   pinning removes per-image illumination freedom.
# => Anchor strength is a clean one-variable tradeoff; sweep it + try a
#    deadzone that decouples force strength from freedom.
#
# R51a: anchor w 0.10                  -> minimal force, maximal freedom
# R51b: anchor w 0.25                  -> middle point
# R51c: anchor w 0.50 + deadzone 0.15  -> full force outside +-15% band only
#
# EXPECTED:
# - no >0.5 dB valley in 5.5-9.5K (else the arm's force is too weak)
# - PSNR >= 22.4 (the ~3K iters r50a wasted in the valley become training)
# - SSIM >= 0.80 (r50c proved reachable), LPIPS <= 0.163

set -e
set -o pipefail  # a python crash through the tee pipe must abort the series

CONFIGS=(
  "Options/ISB_ecaformer_r51a_anchor_w010.yml"
  "Options/ISB_ecaformer_r51b_anchor_w025.yml"
  "Options/ISB_ecaformer_r51c_anchor_deadzone.yml"
)

echo "=========================================="
echo "R51 series: anchor strength sweep + deadzone"
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
echo "R51 series completed."
echo "Check: valley depth 5.5-9.5K per arm, l_anchor + x1_mean_* curves in"
echo "       TensorBoard, PSNR vs the 22.21 champion and r50c's SSIM 0.8064."
echo "=========================================="
