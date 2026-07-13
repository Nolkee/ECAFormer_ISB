#!/bin/bash
# R50 Series: fix the R49 regressions (desaturation + earlier crash) while
# keeping the R49 wins (early images not green, ema_warmup).
#
# DIAGNOSIS (2026-07-14, verified at code level + measured on LOLv1):
# 1. Desaturation = permanent gray-world residual strips the per-image color
#    cast; mapping (behind GroupNorm) can only add a dataset-constant DC, so
#    outputs wash toward the dataset-average cast. Chroma loss (0.05 on a
#    global std scalar) is ~65x too weak to resist.
# 2. Crash moved earlier (5.5-7K vs R48b 7-9.5K) = detached reciprocal WB
#    gains (gray/ch_mean, ch_mean~0.05-0.1) amplify estimator drift into
#    residual-base swings; ema_warmup also surfaces the live-net transient
#    with less smoothing (tau 612-779 vs 1000 during the window).
# 3. R49b anchor was INERT: target mean_c(gt) unreachable (x1 <= 2*lq;
#    2*lq_gray 0.09 << gt_gray 0.49 on 100% of images) -> constant loss,
#    saturated one-sided gradient. Invisible because only l_total was logged.
#
# R50a: gray-world blend decays 1->0 over iters 1500-3500   -> color restored
# R50b: R50a + estimator_lr_mult 0.3 (slow x1 drift)        -> crash fix A
# R50c: R50a + reachable anchor (x1_lq, ratio 1.5, w 0.5)   -> crash fix B
#
# All runs now log every loss component + x1 channel means + WB gains/blend
# to TensorBoard (crash-window drift is directly visible).
#
# EXPECTED:
# - iter-500 baseline still color-neutral (decay starts at 1500)
# - best-checkpoint images: saturation matches GT (no gray wash)
# - r50b/r50c: no 2-3 dB PSNR valley, no post-9K sag
# - targets: PSNR >= 22.4, SSIM >= 0.80, LPIPS <= 0.164

set -e
set -o pipefail  # a python crash through the tee pipe must abort the series

CONFIGS=(
  "Options/ISB_ecaformer_r50a_gw_decay.yml"
  "Options/ISB_ecaformer_r50b_gw_decay_estlr.yml"
  "Options/ISB_ecaformer_r50c_gw_decay_anchor.yml"
)

echo "=========================================="
echo "R50 series: color restore + drift stability"
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
echo "R50 series completed."
echo "Check: best_results/ saturation vs GT, x1_mean_*/gw_* curves in"
echo "       TensorBoard around any PSNR dip, valley depth vs R49/R48b."
echo "=========================================="
