#!/bin/bash
# R53 Series: final pre-AAAI round — hold the r52b record peak, de-luck the
# engage timing, and add a second seed for mean+-std.
#
# R52 VERDICT (2026-07-19):
# - r52b @ 11.5K = NEW ALL-TIME CHAMPION: PSNR 22.689 / SSIM 0.8042 /
#   LPIPS 0.1664 (previous bests: 22.21 PSNR, 0.7964 SSIM at a single
#   checkpoint). The late-anchor / phase-transition design is validated.
# - Problem 1: post-peak sag remains (-0.76 dB from 12.5K to 15.5K) — the
#   0.05 deadzone lets the regime wander ~5% unpenalized. r50c/r51c (tight
#   pins) never sagged -> R53 locks with deadzone 0.
# - Problem 2: transition timing is stochastic (5.5K vs 7.5K on identical
#   configs); r52a's fixed 12K engage fired AFTER the peak and locked a
#   declined state -> R53b engages automatically at the recovery turn.
# - r52c (zero_init_mapping_bias on the anchored base): REFUTED — worse
#   everywhere (21.18/0.786). Direction closed.
#
# R53a: r52b + deadzone 0 + patience 12       -> hold the 22.7 peak
# R53b: r53a + auto engage (+cap 12K, freeze delay 1.5K) -> de-luck the timing
# R53c: r53a with seed 3407                   -> second draw + seed statistics
#
# After picking the winner, run the generalization config for the paper:
#   python -m basicsr.train --opt Options/ISB_ecaformer_r53_lolv2real.yml
#   (verify data/LOLv2Real/Test dir names first)
#
# EXPECTED: peak >= 22.6 that HOLDS (no -0.5 dB sag within patience), SSIM
# >= 0.80 at the same checkpoint, LPIPS continuing to improve on the plateau.

set -e
set -o pipefail  # a python crash through the tee pipe must abort the series

CONFIGS=(
  "Options/ISB_ecaformer_r53a_lock9k_dz0.yml"
  "Options/ISB_ecaformer_r53b_auto_engage.yml"
  "Options/ISB_ecaformer_r53c_lock9k_seed3407.yml"
)

echo "=========================================="
echo "R53 series: hold the peak + de-luck engage"
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
echo "R53 series completed."
echo "Check: does the peak hold past 15K (r52b sagged -0.76 dB)? In the log,"
echo "grep '\[anchor\]' for valley-detected/engaged iters; TensorBoard:"
echo "anchor_engaged + anchor_ratio_ema_* (frozen after engage+delay)."
echo "=========================================="
