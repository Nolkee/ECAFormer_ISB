#!/bin/bash
# R32 Series: Anti-overfitting ablation on R31a (24K ablation)
#
# R31a baseline (250K): PSNR=23.62, SSIM=0.870, LPIPS=0.177
#   train PSNR ~28.45 vs val PSNR ~23.62 → 4.8 dB overfitting gap
#
# R32a: + mixup (mixup_beta=1.2)
# R32b: + weight_decay=0.1
# R32c: + gt_size=192 + dataset_enlarge_ratio=2
# R32d: + tv_loss_weight=0.01 (improve LPIPS)
#
# Common changes for all R32:
#   - loss_on_clamped_output=true (numerical stability)
#   - print_freq=200, output_range_log_interval=500 (less verbose)
#   - early_stop_patience_val=8 (24K needs longer patience)
#   - total_iter=24000 (24K ablation)

set -e

CONFIGS=(
    "Options/ISB_ecaformer_r32a_mixup.yml"
    "Options/ISB_ecaformer_r32b_wd01.yml"
    "Options/ISB_ecaformer_r32c_gt192.yml"
    "Options/ISB_ecaformer_r32d_tv.yml"
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

echo "R32 ablation (24K) completed."
