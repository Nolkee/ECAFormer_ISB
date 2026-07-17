#!/bin/bash
# R52 Series: late self-calibrating anchor (phase-transition-aware) + perceptual push.
#
# R51 VERDICT (2026-07-16) — the tradeoff model was WRONG, replaced by the
# phase-transition model:
# - Every run that reached PSNR 22.2 CRASHED first (R48b, r50a); every run
#   anchored from iter 0 was capped at ~21.8 (r50c 21.75, r51c 21.83) even
#   with a +-15% deadzone -> the mid-training valley is a TRANSITION into a
#   higher-PSNR illumination regime, and an early anchor blocks it.
# - Weak anchors are strictly bad: r51a (w0.1) still crashed (-2.9 dB) and
#   its fixed 1.5*lq target dragged the recovered regime (-0.23 dB); r51b
#   (w0.25) crashed EARLIER and finished last (21.61). No middle sweet spot.
#
# R52a: no anchor until 12K, then lock the ACHIEVED regime (x1_ema frozen)
# R52b: same, engaged at 9K (right after the transition lands)
# R52c: r51c + zero_init_mapping_bias (R41b perceptual trick on a stable base)
#
# EXPECTED:
# - r52a/b: r50a's curve through the transition (valley 6-8K, recovery to
#   ~22.1-22.2), then NO 16K+ sag; targets PSNR >= 22.2 AND SSIM >= 0.80
#   (a point not on the current Pareto front). Watch anchor_ratio_ema_* in
#   TensorBoard: it should jump during the transition and freeze at engage.
# - r52c: no valley, SSIM > 0.805, LPIPS < 0.163 (perceptual record chase).

set -e
set -o pipefail  # a python crash through the tee pipe must abort the series

CONFIGS=(
  "Options/ISB_ecaformer_r52a_late_ema_anchor.yml"
  "Options/ISB_ecaformer_r52b_late_ema_anchor_9k.yml"
  "Options/ISB_ecaformer_r52c_deadzone_zero_bias.yml"
)

echo "=========================================="
echo "R52 series: late EMA anchor + perceptual push"
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
echo "R52 series completed."
echo "Check: r52a/b hold the 22.2 plateau past 16K (r50a sagged -0.43 dB),"
echo "       anchor_ratio_ema_* frozen after engage, r52c SSIM/LPIPS records."
echo "=========================================="
